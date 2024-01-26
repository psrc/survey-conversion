# Copyright [2023] [Puget Sound Regional Council]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import datetime
import toml
import urllib
import pyodbc
import pandas as pd
import sqlalchemy
from sqlalchemy.engine import URL
import logging
from daysim import logcontroller
from modules import util, convert

pd.options.mode.chained_assignment = None  # default='warn'

config = toml.load("daysim_configuration.toml")

# Start log file
logger = logcontroller.setup_custom_logger("convert_format_logger.txt")
logger.info("--------------------convert_format.py STARTING--------------------")
start_time = datetime.datetime.now()

def total_persons_to_hh(
    hh,
    person,
    daysim_field,
    filter_field,
    filter_field_list,
    hhid_col="hhno",
    wt_col="hhexpfac",
):
    """Use person field to calculate total number of person in a household for a given field
    e.g., total number of full-time workers"""

    df = person[person[filter_field].isin(filter_field_list)]
    df = df.groupby(hhid_col).count().reset_index()[[wt_col, hhid_col]]
    df.rename(columns={wt_col: daysim_field}, inplace=True)

    hh.drop(daysim_field, inplace=True, axis=1)

    # Join to households
    hh = pd.merge(hh, df, how="left", on=hhid_col)
    hh[daysim_field].fillna(0, inplace=True)

    return hh


def process_household_file(hh, person):
    """ Convert household columns and generate data from person table"""

    expr_df = pd.read_csv(r"daysim\inputs\hh_expr_daysim.csv")

    # Apply expression file input
    hh = convert.process_expression_file(hh, expr_df, config['hh_columns'])

    # Workers in Household from Person File (hhwkrs)
    hh = total_persons_to_hh(
        hh,
        person,
        daysim_field="hhwkrs",
        filter_field="pwtyp",
        filter_field_list=[1, 2],
        hhid_col="hhno",
        wt_col="psexpfac",
    )

    # Workers by type in household
    for hh_field, pwtyp_filter in config["pwtyp_map"].items():
        hh = total_persons_to_hh(
            hh,
            person,
            daysim_field=hh_field,
            filter_field="pwtyp",
            filter_field_list=[pwtyp_filter],
            hhid_col="hhno",
            wt_col="psexpfac",
        )

    # Person by type in houseohld
    for hh_field, pptyp_filter in config["pptyp_map"].items():
        hh = total_persons_to_hh(
            hh,
            person,
            daysim_field=hh_field,
            filter_field="pptyp",
            filter_field_list=[pptyp_filter],
            hhid_col="hhno",
            wt_col="psexpfac",
        )

    # Remove households without parcels
    _filter = -hh["hhparcel"].isnull()
    logger.info(f"Dropped {len(hh[~_filter])} households: missing parcels")
    hh = hh[_filter]

    return hh


def process_trip_file(df, person):
    """Convert trip records to Daysim format."""

    # Trips to/from work are considered "usual workplace" only if dtaz == workplace TAZ
    # must join person records to get usual work and school location
    df = df.merge(
        person[["person_id", "pno", "pwpcl", "pspcl", "pwtaz", "pstaz"]],
        how="left",
        on="person_id",
    )

    # Calculate fields using an expression file
    expr_df = pd.read_csv(r"daysim\inputs\trip_expr_daysim.csv")

    # Select only weekday trips (M-Th)
    df = convert.apply_filter(
        df,
        "trips",
        df["dayofweek"].isin(["Monday", "Tuesday", "Wednesday", "Thursday"]),
        logger,
        "trip taken on Friday, Saturday, or Sunday",
    )

    df = convert.apply_filter(
        df,
        "trips",
        (-df["arrival_time_timestamp"].isnull())
        & (-df["depart_time_timestamp"].isnull()),
        logger,
        "missing departure/arrival time",
    )

    ##############################
    # Start and end time
    ##############################

    # The time_mam (minutes after midnight) data seems to be corrupted; recalculate from the timestamp
    df["arrive_hour"] = (
        df["arrival_time_timestamp"]
        .apply(lambda x: str(x).split(" ")[-1].split(":")[0])
        .astype("int")
    )
    df["arrive_min"] = (
        df["arrival_time_timestamp"]
        .apply(lambda x: str(x).split(" ")[-1].split(":")[1])
        .astype("int")
    )
    df["arrtm"] = df["arrive_hour"] * 60 + df["arrive_min"]

    df["depart_hour"] = (
        df["depart_time_timestamp"]
        .apply(lambda x: str(x).split(" ")[-1].split(":")[0])
        .astype("int")
    )
    df["depart_min"] = (
        df["depart_time_timestamp"]
        .apply(lambda x: str(x).split(" ")[-1].split(":")[1])
        .astype("int")
    )

    df["deptm"] = df["depart_hour"] * 60 + df["depart_min"]    

    # Calculate start of next trip (ENDACTTM: trip destination activity end time)
    # FIXME: there are negative values in the activity_duration field
    # Note: this field name will change for 2023. It was a derived field for 2017
    # and not calculated for 2019.
    df["endacttm"] = df["activity_duration"].abs() + df["arrtm"]

     # Daysim-specific need: add pathtype by analyzing transit submode
    # FIXME: Note that this field doesn't exist for some trips, should really be analyzed by grouping on the trip day or tour
    df["pathtype"] = 1
    for index, row in df.iterrows():
        if len(
            [
                i
                for i in list(row[["mode_1", "mode_2", "mode_3", "mode_4"]].values)
                if i in config["transit_mode_list"]
            ]
        ):
            # ferry or water taxi
            if (
                "Ferry or water taxi"
                in row[["mode_1", "mode_2", "mode_3", "mode_4"]].values
            ):
                df.loc[index, "pathtype"] = 7
            # commuter rail
            elif (
                "Commuter rail (Sounder, Amtrak)"
                in row[["mode_1", "mode_2", "mode_3", "mode_4"]].values
            ):
                df.loc[index, "pathtype"] = 6
            # 'Urban rail (e.g., Link light rail, monorail)'
            elif [
                "Urban Rail (e.g., Link light rail, monorail)"
                or "Other rail (e.g., streetcar)"
            ] in row[["mode_1", "mode_2", "mode_3", "mode_4"]].values:
                df.loc[index, "pathtype"] = 4
            else:
                df.loc[index, "pathtype"] = 3

            # FIXME!!!
            # Note that we also need to include KnR and TNC?

    df = convert.process_expression_file(df, expr_df, config['trip_columns'])

    # Filter out trips that started before 0 minutes after midnight
    df = convert.apply_filter(
        df,
        "trips",
        df["deptm"] >= 0,
        logger,
        "trips started before 0 minutes after midnight",
    )

    df = convert.apply_filter(
        df,
        "trips",
        (-df["trexpfac"].isnull()) & (df["trexpfac"] > 0),
        logger,
        "no or null weight",
    )

    for col in ["opurp", "dpurp"]:
        df = convert.apply_filter(
            df, "trips", (df[col] >= 0), logger, "missing or unusable " + col
        )


    # if arrtm/deptm > 24*60, subtract that value to normalize to a single day
    for colname in ["arrtm", "deptm"]:
        for i in range(2, int(np.ceil(df[colname] / (24 * 60)).max()) + 1):
            filter = (df[colname] > (24 * 60)) & (df[colname] < (24 * 60) * i)
            df.loc[filter, colname] = df.loc[filter, colname] - 24 * 60 * (i - 1)

    df = convert.apply_filter(
        df,
        "trips",
        ~(
            (df["otaz"] == df["dtaz"])
            & (df["opurp"] == df["dpurp"])
            & (df["opurp"] == 1)
        ),
        logger,
        "intrazonal work-related trips",
    )

    # Filter null trips
    for col in ["mode", "opurp", "dpurp", "otaz", "dtaz"]:
        df = convert.apply_filter(df, "trips", -df[col].isnull(), logger, col + " is null")

    return df


def build_tour_file(trip, person):
    """Generate tours from Daysim-formatted trip records by iterating through person-days."""

    # Keep track of error types
    error_dict = {
        "first O and last D are not home": 0,
        "different number of tour starts and ends at home": 0,
        "dpurp of previous trip does not match opurp of next trip": 0,
        "activity type of previous trip does not match next trip": 0,
        "different number of tour starts/ends at home": 0,
        "no trips in set": 0,
        "no purposes provided except change mode": 0
    }

    trip = convert.apply_filter(
        trip,
        "trips",
        -((trip["opurp"] == trip["dpurp"]) & (trip["opurp"] == 0)),
        logger,
        "trips have same origin/destination of home",
    )

    trip = convert.apply_filter(
        trip,
        "trips",
        (((trip["dpurp"] >= 0)) | (trip["opurp"] >= 0)),
        logger,
        "trips missing purpose",
    )

    # Reset the index
    trip = trip.reset_index()

    tour_dict = {}
    bad_trips = []
    tour_id = 1
    
    if len(config['debug_person_id']) > 0:
        person_id_list = config['debug_person_id']
    else:
        # Get all unique person ID values
        person_id_list = trip["unique_person_id"].value_counts().index.values

    for person_id in person_id_list:
        print(str(tour_id))
        person_df = trip.loc[trip["unique_person_id"] == person_id]

        # Loop through each travel day
        for day in person_df["day"].unique():
            df = person_df.loc[person_df["day"] == day]

            # First O and last D of person's travel day should be home; if not, skip this trip set
            # FIXME: consider keeping other trips that conform
            if (
                df.groupby("unique_person_id").first()["opurp"].values[0] != 0
            ) or df.groupby("unique_person_id").last()["dpurp"].values[0] != 0:
                bad_trips += df["trip_id"].tolist()
                error_dict["first O and last D are not home"] += 1
                continue

            # Some people do not report sequential trips.
            # If the dpurp of the previous trip does not match the opurp of the next trip, skip...
            df["next_opurp"] = df.shift(-1)[["opurp"]]
            df["prev_opurp"] = df.shift(1)[["opurp"]]
            df["next_oadtyp"] = df.shift(-1)[["oadtyp"]]
            if (
                len(df.iloc[:-1][(df.iloc[:-1]["next_opurp"] != df.iloc[:-1]["dpurp"])])
                > 0
            ):
                bad_trips += df["trip_id"].tolist()
                error_dict[
                    "dpurp of previous trip does not match opurp of next trip"
                ] += 1
                continue

            # Apply similarly for activity type
            if (
                len(
                    df.iloc[:-1][
                        (df.iloc[:-1]["next_oadtyp"] != df.iloc[:-1]["dadtyp"])
                    ]
                )
                > 0
            ):
                bad_trips += df["trip_id"].tolist()
                error_dict[
                    "activity type of previous trip does not match next trip"
                ] += 1
                continue

            # Identify home-based tours
            home_tours_start = df[df["opurp"] == 0]
            home_tours_end = df[df["dpurp"] == 0]

            # Skip person if they have a different number of tour starts/ends at home
            if len(home_tours_start) != len(home_tours_end):
                bad_trips += df["trip_id"].tolist()
                error_dict["different number of tour starts/ends at home"] += 1
                continue

            ########################################
            # Process Each Home-Based Tour 
            ########################################
            # Loop through each of the home-based tours identified above
            for tour_start_index in range(len(home_tours_start)):
                tour_dict[tour_id] = {}

                # Iterate between the first and last row of trips to build the tour
                start_row_id = home_tours_start.index[tour_start_index]
                end_row_id = home_tours_end.index[tour_start_index]

                # Select slice of trips that represent a set of home-to-home trips
                _df = df.loc[start_row_id:end_row_id]

                if len(_df) == 0:
                    bad_trips += _df["trip_id"].tolist()
                    error_dict["no trips in set"] += 1
                    continue

                # If no other trip purposes besides home and change mode, skip
                if _df.loc[_df['dpurp'] != 0,'dpurp'].unique()[0] == 10:
                    bad_trips += _df["trip_id"].tolist()
                    error_dict["no purposes provided except change mode"] += 1
                    continue

                # Calculate duration at location between each trip
                # (difference between arrival at a destination and start of next trip)
                _df["duration"] = (
                    _df.shift(-1).iloc[:-1]["deptm"] - _df.iloc[:-1]["arrtm"]
                )

                ########################################
                # Check for Subtours
                ########################################
                # Keep track of subtours for ID indexing
                subtour_count = 0

                # Check if we are dealing with work-based subtours
                # There must be 4+ trips for a subtour to exist; usual workplace destination appears at least twice (oadtyp==2)
                # Ensure we don't capture return trips home by requiring destination purpose of usualy workplace and non-home types
                if (
                    (len(_df) >= 4)
                    & (len(_df[_df["oadtyp"] == 2]) >= 2)
                    & (len(_df[_df["opurp"] == 1]) >= 2)
                    & (len(_df[(_df["oadtyp"] == 2) & (_df["dadtyp"] > 2)]) >= 1)
                ):
                    # Potential that subtours exist; test each one to see if it's usable
                    subtour_index_start_values = _df[
                        (_df["oadtyp"] == 2) & (_df["dadtyp"] > 2)
                    ].index.values
                    subtours_df = pd.DataFrame()

                    # Loop through each potential subtour
                    # the following trips must eventually return to work for this to qualify as a subtour
                 
                    ########################################
                    # Process Subtours
                    ########################################
                    for subtour_start_value in subtour_index_start_values:
                        # Potential subtour
                        # Loop through the index from subtour start
                        # Create a new subtour ID; subtour_count is 0-based so add 1 to increment
                        subtour_id = tour_id + subtour_count + 1

                        next_row_index_start = (
                            np.where(_df.index.values == subtour_start_value)[0][0]
                            + 1
                        )
                        for i in _df.index.values[next_row_index_start:]:
                            next_row = _df.loc[i]
                            # If next trip is to usual workplace, this marks the end of the subtour
                            if next_row["dadtyp"] == 2:
                                subtour_df = _df.loc[subtour_start_value:i]

                                # local_tour_id += 1

                                tour_dict[subtour_id] = {}
                                # Process this subtour
                                # Create a new tour record for the subtour
                                subtour_df["subtour_id"] = subtour_id
                                subtours_df = subtours_df.append(subtour_df)

                                # Subtours are added as separate tour records
                                tour_dict[subtour_id]["tour"] = subtour_id

                                tour_dict = convert.add_tour_data(subtour_df, tour_dict, subtour_id, day)

                                tour_dict[subtour_id][
                                    "parent"
                                ] = tour_id  # Parent is the main tour ID
                                tour_dict[subtour_id][
                                    "subtrs"
                                ] = 0  # No subtours for subtours

                                trip = convert.update_trip_data(trip, subtour_df, subtour_id)

                                subtour_count += 1

                                break
                            else:
                                continue

                    # No subtours actually found, treat as regular set of trips
                    if len(subtours_df) < 1:
                        # No subtours actually found
                        # FIXME: make this a function, because it's called multiple times
                        tour_dict[tour_id]["subtrs"] = 0
                        tour_dict[tour_id]["parent"] = 0

                        tour_dict = convert.add_tour_data(_df, tour_dict, tour_id, day)

                        trip = convert.update_trip_data(trip, _df, tour_id)

                    else:
                        # Subtours were identified
                        # Fill out primary tour and trip data
                        for col in ["hhno","household_id_elmer","pno","person_id","unique_person_id"]:
                                tour_dict[tour_id][col] = _df.iloc[0][col]

                        tour_dict[tour_id]["day"] = day
                        # First trip row contains departure time and origin info
                        tour_dict[tour_id]["tlvorig"] = df.iloc[0]["deptm"]
                        tour_dict[tour_id]["totaz"] = df.iloc[0]["otaz"]
                        tour_dict[tour_id]["topcl"] = df.iloc[0]["opcl"]
                        tour_dict[tour_id]["toadtyp"] = df.iloc[0]["oadtyp"]

                        # Last trip row contains return info
                        tour_dict[tour_id]["tarorig"] = df.iloc[-1]["arrtm"]

                        # The main tour destination arrival will be the trip before subtours
                        # the main tour destination departure will be the trip after subtours
                        # trip when they arrive to work -> always the previous trip before subtours_df index begins
                        main_tour_start_index = _df.index.values[
                            np.where(_df.index.values == subtours_df.index[0])[0][0]
                            - 1
                        ]
                        # trip when leave work -> always the next trip after the end of the subtours_df
                        main_tour_end_index = _df.index.values[
                            np.where(_df.index.values == subtours_df.index[-1])[0][
                                0
                            ]
                            + 1
                        ]
                        # If there were subtours, this is a work tour
                        tour_dict[tour_id]["pdpurp"] = 1
                        tour_dict[tour_id]["tdtaz"] = _df.loc[
                            main_tour_start_index
                        ]["dtaz"]
                        tour_dict[tour_id]["tdpcl"] = _df.loc[
                            main_tour_start_index
                        ]["dpcl"]
                        tour_dict[tour_id]["tdadtyp"] = _df.loc[
                            main_tour_start_index
                        ]["dadtyp"]

                        # Pathtype is defined by a heirarchy, where highest number is chosen first
                        # Ferry > Commuter rail > Light Rail > Bus > Auto Network
                        # Note that tour pathtype is different from trip path type (?)
                        subtours_excluded_df = pd.concat(
                            [
                                df.loc[start_row_id:main_tour_start_index],
                                df.loc[main_tour_end_index:end_row_id],
                            ]
                        )
                        tour_dict[tour_id]["tpathtp"] = subtours_excluded_df.loc[
                            subtours_excluded_df["mode"].idxmax()
                        ]["pathtype"]

                        # Calculate tour halves, etc
                        tour_dict[tour_id]["tripsh1"] = len(
                            _df.loc[0:main_tour_start_index]
                        )
                        tour_dict[tour_id]["tripsh2"] = len(
                            _df.loc[main_tour_end_index:]
                        )

                        # Set tour halves on trip records
                        trip.loc[
                            trip["trip_id"].isin(
                                _df.loc[0:main_tour_start_index].trip_id
                            ),
                            "half",
                        ] = 1
                        trip.loc[
                            trip["trip_id"].isin(
                                _df.loc[main_tour_end_index:].trip_id
                            ),
                            "half",
                        ] = 2

                        # set trip segment within half tours
                        trip.loc[
                            trip["trip_id"].isin(
                                _df.loc[0:main_tour_start_index].trip_id
                            ),
                            "tseg",
                        ] = range(1, len(_df.loc[0:main_tour_start_index]) + 1)
                        trip.loc[
                            trip["trip_id"].isin(
                                _df.loc[main_tour_end_index:].trip_id
                            ),
                            "tseg",
                        ] = range(1, len(_df.loc[main_tour_end_index:]) + 1)

                        # Departure/arrival times
                        tour_dict[tour_id]["tlvdest"] = _df.loc[
                            main_tour_end_index
                        ]["deptm"]
                        tour_dict[tour_id]["tardest"] = _df.loc[
                            main_tour_start_index
                        ]["arrtm"]

                        # Number of subtours
                        tour_dict[tour_id]["subtrs"] = subtour_count
                        tour_dict[tour_id]["parent"] = 0

                        # Mode
                        tour_dict[tour_id]["tmodetp"] = convert.assign_tour_mode(
                            _df, tour_dict, tour_id
                        )

                        tour_dict[tour_id]['tour'] = tour_id

                        # add tour ID to the trip records (for trips not in the subtour_df)
                        df_unique_no_subtours = [
                            i
                            for i in _df["trip_id"].values
                            if i not in subtours_df["trip_id"].values
                        ]
                        df_unique_no_subtours = _df[
                            _df["trip_id"].isin(df_unique_no_subtours)
                        ]
                        trip.loc[
                            trip["trip_id"].isin(
                                df_unique_no_subtours["trip_id"].values
                            ),
                            "tour",
                        ] = tour_id

                else:
                    # No subtours
                    # Add tour data and update trip data
                    tour_dict = convert.add_tour_data(_df, tour_dict, tour_id, day)
                    # No subtours 
                    tour_dict[tour_id]["subtrs"] = 0
                    tour_dict[tour_id]["parent"] = 0

                    tour_dict[tour_id]['tour'] = tour_id

                    trip = convert.update_trip_data(trip, _df, tour_id)

                tour_id = tour_id + subtour_count + 1

    tour = pd.DataFrame.from_dict(tour_dict, orient="index")

    # After tour file is created, apply expression file for tours
    expr_df = pd.read_csv(r"daysim\inputs\tour_expr_daysim.csv")

    tour = convert.process_expression_file(tour, expr_df, config["tour_columns"])

    # Assign weight toexpfac as hhexpfac (getting it from psexpfac, which is the same as hhexpfac)
    tour = tour.merge(
        person[["unique_person_id", "psexpfac"]], on="unique_person_id", how="left"
    )
    tour.rename(columns={"psexpfac": "toexpfac"}, inplace=True)

    # remove the trips that weren't included in the tour file
    _filter = -trip["trip_id"].isin(bad_trips)
    logger.info(f"Dropped {len(trip[~_filter])} total trips due to tour issues ")
    trip = trip[_filter]
    pd.DataFrame(bad_trips).T.to_csv(
        os.path.join(config["output_dir"], "bad_trips.csv")
    )

    trip = trip[config['trip_columns'] + ["unique_person_id","household_id_elmer"]]

    return tour, trip, error_dict


def process_household_day(tour, hh):
    household_day = tour.groupby(["hhno", "day"]).count().reset_index()[["hhno", "day"]]

    # add day of week lookup
    household_day["dow"] = household_day["day"]

    # Set number of joint tours to 0 for this version of Daysim
    for col in ["jttours", "phtours", "fhtours"]:
        household_day[col] = 0

    # Add expansion factor
    household_day = household_day.merge(hh[["hhno", "hhexpfac"]], on="hhno", how="left")
    household_day.rename(columns={"hhexpfac": "hdexpfac"}, inplace=True)

    return household_day


def process_person_day(tour, person, trip, hh, person_day_original_df):
    # Get the usual workplace column from person records
    tour = tour.merge(person[["hhno", "pno", "pwpcl"]], on=["hhno", "pno"], how="left")

    pday = pd.DataFrame()
    for person_rec in person["unique_person_id"].unique():
        # get this person's tours
        _tour = tour[tour["unique_person_id"] == person_rec]

        if len(_tour) > 0:
            # Loop through each day
            for day in _tour["day"].unique():
                day_tour = _tour[_tour["day"] == day]

                # prec_id = str(person_rec) + str(day)
                pday.loc[person_rec, "hhno"] = day_tour["hhno"].iloc[0]
                pday.loc[person_rec, "pno"] = day_tour["pno"].iloc[0]
                pday.loc[person_rec, "person_id"] = day_tour["person_id"].iloc[0]
                pday.loc[person_rec, "household_id_elmer"] = day_tour[
                    "household_id_elmer"
                ].iloc[0]
                pday.loc[person_rec, "unique_person_id"] = person_rec
                pday.loc[person_rec, "day"] = day
                pday.loc[person_rec, "pdexpfac"] = person[
                    person["unique_person_id"] == person_rec
                ].iloc[0]["psexpfac"]

                # Begin/End at home-
                # need to get from first and last trips of tour days
                pday.loc[person_rec, "beghom"] = 0
                pday.loc[person_rec, "endhom"] = 0
                _trip = trip[
                    (trip["unique_person_id"] == person_rec) & (trip["day"] == day)
                ]
                if _trip.iloc[0]["opurp"] == 0:
                    pday.loc[person_rec, "beghom"] = 1
                if _trip.iloc[-1]["dpurp"] == 0:
                    pday.loc[person_rec, "endhom"] = 1

                # Number of tours by purpose
                for purp_name, purp_val in config["purp_dict"].items():
                    pday.loc[person_rec, purp_name + "tours"] = len(
                        day_tour[day_tour["pdpurp"] == int(purp_val)]
                    )

                    # Number of stops
                    day_tour_purp = day_tour[day_tour["pdpurp"] == int(purp_val)]
                    if len(day_tour_purp) > 0:
                        nstops = day_tour_purp[["tripsh1", "tripsh2"]].sum().sum() - 2
                    else:
                        nstops = 0
                    pday.loc[person_rec, purp_name + "stops"] = nstops

                # Home based tours
                pday.loc[person_rec, "hbtours"] = len(day_tour)

                # Work-based tours (subtours)
                pday.loc[person_rec, "wbtours"] = day_tour["subtrs"].sum()

                # Work trips to usual workplace
                pday.loc[person_rec, "uwtours"] = len(
                    day_tour[day_tour["tdpcl"] == day_tour["pwpcl"]]
                )

    # Calculate work at home hours
    person_day_original_df["telework_hr"] = (
        person_day_original_df["telework_time"]
        .astype("str")
        .apply(lambda x: x.split("hour")[0])
    )
    person_day_original_df["telework_hr"] = (
        pd.to_numeric(person_day_original_df["telework_hr"], errors="coerce")
        .fillna(0)
        .astype("int")
    )
    person_day_original_df["telework_min"] = (
        person_day_original_df["telework_time"]
        .astype("str")
        .apply(lambda x: x.split("hour")[-1])
    )
    person_day_original_df["telework_min"] = person_day_original_df[
        "telework_min"
    ].apply(lambda x: x.split("minutes")[0])
    person_day_original_df["telework_min"] = person_day_original_df[
        "telework_min"
    ].apply(lambda x: x.split("s ")[-1])
    person_day_original_df["telework_min"] = (
        pd.to_numeric(person_day_original_df["telework_min"], errors="coerce")
        .fillna(0)
        .astype("int")
    )
    person_day_original_df["wkathome"] = (
        person_day_original_df["telework_hr"]
        + person_day_original_df["telework_min"] / 60.0
    )

    # Add work at home from Person Day Elmer file
    pday["household_id_elmer"] = pday["household_id_elmer"].astype("int")
    pday = pday.merge(
        person_day_original_df[["household_id", "pernum", "day", "wkathome"]],
        left_on=["household_id_elmer", "pno", "day"],
        right_on=["household_id", "pernum", "day"],
        how="inner",
    )

    # Add person day records of no travel
    # Get the unique person ID to merge with person records
    person_day_original_df = person_day_original_df.merge(
        hh[["household_id_elmer", "hhno"]],
        left_on="household_id",
        right_on="household_id_elmer",
    )
    person_day_original_df["unique_person_id"] = person_day_original_df["hhno"].astype(
        "str"
    ) + person_day_original_df["pernum"].astype("str")
    no_travel_df = person_day_original_df[person_day_original_df["numtrips"] == 0]
    no_travel_df = no_travel_df[
        no_travel_df["unique_person_id"].isin(person["unique_person_id"])
    ]
    # Only add entries for completed survey days
    no_travel_df = no_travel_df[no_travel_df["svy_complete"] == "Complete"]

    pday["no_travel_flag"] = 0

    for person_rec in no_travel_df.unique_person_id.unique():
        pday.loc[person_rec, :] = 0
        pday.loc[person_rec, "no_travel_flag"] = 1
        pday.loc[person_rec, "hhno"] = no_travel_df[
            no_travel_df["unique_person_id"] == person_rec
        ]["hhno"].values[0]
        pday.loc[person_rec, "pno"] = no_travel_df[
            no_travel_df["unique_person_id"] == person_rec
        ]["pernum"].values[0]
        pday.loc[person_rec, "person_id"] = no_travel_df[
            no_travel_df["unique_person_id"] == person_rec
        ]["person_id"].values[0]
        pday.loc[person_rec, "unique_person_id"] = person_rec
        pday.loc[
            person_rec, "day"
        ] = 1  # these will all be replaced after this step so set to 1
        pday.loc[person_rec, "beghom"] = 1
        pday.loc[person_rec, "endhom"] = 1
        pday.loc[person_rec, "wkathome"] = no_travel_df[
            no_travel_df["unique_person_id"] == person_rec
        ]["wkathome"].values[0]
        pday.loc[person_rec, "pdexpfac"] = person[
            person["unique_person_id"] == person_rec
        ]["psexpfac"].values[0]

    return pday


def convert_format():

    # Load Person Day data from Elmer
    person_day_original_df = util.load_elmer_table("HHSurvey.v_days", config["survey_year"])
    person_day_original_df["day"] = person_day_original_df["dayofweek"].map(
        config["day_map"]
    )

    if not config["debug_tours"]:
        # Load geolocated survey data
        trip_original_df = pd.read_csv(
            os.path.join(config["input_dir"], "geolocated_trip.csv")
        )
        hh_original_df = pd.read_csv(
            os.path.join(config["input_dir"], "geolocated_hh.csv")
        )
        person_original_df = pd.read_csv(
            os.path.join(config["input_dir"], "geolocated_person.csv")
        )


        # Load expression files
        person_expr_df = pd.read_csv(r"daysim/inputs/person_expr_daysim.csv", delimiter=",(?![^\(]*[\)])")   # Exclude quotes within data
        hh_expr_df = pd.read_csv(r"daysim/inputs/hh_expr_daysim.csv")

        # Recode person, household, and trip data
        person = convert.process_expression_file(person_original_df,
                                                 person_expr_df, config['person_columns'])
        hh = process_household_file(hh_original_df, person)
        # hh = convert.process_expression_file(hh_original_df, hh_expr_df, config["hh_columns"])

        # FIXME: move this to a function or something
        # trip = convert.process_expression_file(trip, trip_expr_df, config['trip_columns'])
        trip = process_trip_file(trip_original_df, person)

        # Write mapping between original trip_id and tsvid used on they survey
        trip[["trip_id", "tsvid"]].to_csv(
            os.path.join(config["output_dir"], "trip_id_tsvid_mapping.csv")
        )

        # Create new Household and Person records for each travel day.
        # For trip/tour models we use this data as if each person-day were independent for multiple-day diaries
        hh["household_id_elmer"] = hh["hhno"].copy()
        trip["household_id_elmer"] = trip["hhno"].copy()
        person["household_id_elmer"] = person["hhno"].copy()

        for day in trip["day"].unique():
            day = int(day)
            trip.loc[trip["day"] == day, "new_hhno"] = (
                trip["hhno"].astype("int") * 10 + day
            ).astype("int")

            hh_day = hh[
                hh["household_id_elmer"].isin(
                    trip.loc[trip["day"] == day, "household_id_elmer"]
                )
            ].copy()
            hh_day["new_hhno"] = (hh_day["hhno"].astype("int") * 10 + day).astype("int")
            hh_day["flag"] = 1
            hh = hh.append(hh_day)
            # Only keep the renamed multi-day households and persons

            person_day = person[
                person["household_id_elmer"].isin(
                    trip.loc[trip["day"] == day, "household_id_elmer"]
                )
            ].copy()
            person_day["new_hhno"] = (
                person_day["hhno"].astype("int") * 10 + day
            ).astype("int")
            person_day["flag"] = 1
            person = person.append(person_day)

        # Remove duplicates of the original cloned households and persons
        person = person[person["flag"] == 1]
        hh = hh[hh["flag"] == 1]

        hh = hh[~hh.duplicated()]
        person = person[~person.duplicated()]

        person.drop("hhno", axis=1, inplace=True)
        hh.drop("hhno", axis=1, inplace=True)
        trip.drop("hhno", axis=1, inplace=True)

        person.rename(columns={"new_hhno": "hhno"}, inplace=True)
        hh.rename(columns={"new_hhno": "hhno"}, inplace=True)
        trip.rename(columns={"new_hhno": "hhno"}, inplace=True)

        if config["write_debug_files"] == True:
            # Temporarily write file to disk so we can reload for debugging
            person.to_csv(os.path.join(config["output_dir"], "daysim_person.csv"))
            hh.to_csv(os.path.join(config["output_dir"], "daysim_hh.csv"))
            trip.to_csv(os.path.join(config["output_dir"], "daysim_trip.csv"))
    else:
        person = pd.read_csv(os.path.join(config["output_dir"], "daysim_person.csv"))
        hh = pd.read_csv(os.path.join(config["output_dir"], "daysim_hh.csv"))
        trip = pd.read_csv(os.path.join(config["output_dir"], "daysim_trip.csv"))

    # Make sure trips are properly ordered, where deptm is increasing for each person's travel day
    trip["person_id_int"] = trip["person_id"].astype("int")
    trip = trip.sort_values(["person_id_int", "day", "deptm"])
    trip = trip.reset_index()

    # Use unique person ID since the Elmer person_id has been copied for multiple days
    trip["unique_person_id"] = trip["hhno"].astype("int").astype("str") + trip[
        "pno"
    ].astype("str")
    person["unique_person_id"] = person["hhno"].astype("int").astype("str") + person[
        "pno"
    ].astype("str")

    # Create tour file and update the trip file with tour info
    tour, trip, error_dict = build_tour_file(trip, person)
    tour["unique_person_id"] = tour["hhno"].astype("int").astype("str") + tour[
        "pno"
    ].astype("str")

    household_day = process_household_day(tour, hh)

    error_dict_df = pd.DataFrame(
        error_dict.values(), index=error_dict.keys(), columns=["errors"]
    )
    for index, row in error_dict_df.iterrows():
        logger.info(f"Dropped {row.errors} tours: " + index)

    # person day
    person_day = process_person_day(tour, person, trip, hh, person_day_original_df)

    # Set all travel days to 1
    trip["day"] = 1
    tour["day"] = 1
    household_day["day"] = 1
    household_day["dow"] = 1
    person_day["day"] = 1

    trip[["travdist", "travcost", "travtime"]] = "-1.00"

    for df_name, df in {
        "_person": person,
        "_trip": trip,
        "_tour": tour,
        "_household": hh,
        "_household_day": household_day,
        "_person_day": person_day,
    }.items():
        print(df_name)

        df.to_csv(
            os.path.join(config["output_dir"], df_name + ".tsv"), index=False, sep="\t"
        )

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------convert_format.py ENDING--------------------")
    logger.info("convert_format.py RUN TIME %s" % str(elapsed_total))
