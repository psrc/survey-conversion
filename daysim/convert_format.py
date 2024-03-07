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
from modules import util, convert, tours, days

pd.options.mode.chained_assignment = None  # default='warn'


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
    e.g., total number of full-time workers. This is a Daysim-specific need."""

    df = person[person[filter_field].isin(filter_field_list)]
    df = df.groupby(hhid_col).count().reset_index()[[wt_col, hhid_col]]
    df.rename(columns={wt_col: daysim_field}, inplace=True)

    if daysim_field in hh.columns:
        hh.drop(daysim_field, inplace=True, axis=1)

    # Join to households
    hh = pd.merge(hh, df, how="left", on=hhid_col)
    hh[daysim_field].fillna(0, inplace=True)

    return hh


def process_household_file(hh, person, df_lookup, config, logger):
    """Convert household columns and generate data from person table"""

    expr_df = pd.read_csv(os.path.join(config["input_dir"], "hh_expr.csv"))

    # Apply expression file input
    hh = convert.process_expression_file(
        hh, expr_df, df_lookup[df_lookup["table"] == "household"]
    )

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


def process_trip_file(df, person, day, df_lookup, config, logger):
    """Convert trip records to Daysim format."""

    # Trips to/from work are considered "usual workplace" only if dtaz == workplace TAZ
    # must join person records to get usual work and school location
    df = df.merge(
        person[["person_id", "pno", "pwpcl", "pspcl", "pwtaz", "pstaz"]],
        how="left",
        on="person_id",
    )

    # Calculate fields using an expression file
    expr_df = pd.read_csv(os.path.join(config["input_dir"], "trip_expr.csv"))

    # Select only weekday trips (M-Th)
    df = convert.apply_filter(
        df,
        "trips",
        df["travel_dow_label"].isin(["Monday", "Tuesday", "Wednesday", "Thursday"]),
        logger,
        "trip taken on Friday, Saturday, or Sunday",
    )

    df = convert.apply_filter(
        df,
        "trips",
        (-df["arrival_time_hour"].isnull()) & (-df["depart_time_hour"].isnull()),
        logger,
        "missing departure/arrival time",
    )

    # Start and end time (in minutes after midnight)
    df["arrtm"] = (df["arrival_time_hour"] * 60) + df["arrival_time_minute"]
    df["deptm"] = (df["depart_time_hour"] * 60) + df["depart_time_minute"]

    # Calculate start of next trip (ENDACTTM: trip destination activity end time)
    df["endacttm"] = df["duration_minutes"] + df["arrtm"]

    # Daysim-specific need: add pathtype by analyzing transit submode
    # FIXME: Note that this field doesn't exist for some trips, should really be analyzed by grouping on the trip day or tour
    # FIXME: this is slow, vectorize
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

    df = convert.process_expression_file(
        df, expr_df, df_lookup[df_lookup["table"] == "trip"]
    )

    # Filter out trips that started before 0 minutes after midnight
    # df = convert.apply_filter(
    #     df,
    #     "trips",
    #     df["deptm"] >= 0,
    #     logger,
    #     "trips started before 0 minutes after midnight",
    # )

    df = convert.apply_filter(
        df,
        "trips",
        (-df["trexpfac"].isnull()) & (df["trexpfac"] > 0),
        logger,
        "no or null weight",
    )

    # for col in ["opurp", "dpurp"]:
    #     df = convert.apply_filter(
    #         df, "trips", (df[col] >= 0), logger, "missing or unusable " + col
    #     )

    # FIXME: this should happen as a catch later; if arrtm/deptm > 24*60, flag it

    # # if arrtm/deptm > 24*60, subtract that value to normalize to a single day
    # for colname in ["arrtm", "deptm"]:
    #     for i in range(2, int(np.ceil(df[colname] / (24 * 60)).max()) + 1):
    #         filter = (df[colname] > (24 * 60)) & (df[colname] < (24 * 60) * i)
    #         df.loc[filter, colname] = df.loc[filter, colname] - 24 * 60 * (i - 1)

    # df = convert.apply_filter(
    #     df,
    #     "trips",
    #     ~(
    #         (df["otaz"] == df["dtaz"])
    #         & (df["opurp"] == df["dpurp"])
    #         & (df["opurp"] == 1)
    #     ),
    #     logger,
    #     "intrazonal work-related trips",
    # )

    # # Filter null trips
    # for col in ["mode", "opurp", "dpurp", "otaz", "dtaz"]:
    #     df = convert.apply_filter(
    #         df, "trips", -df[col].isnull(), logger, col + " is null"
    #     )

    return df


def build_tour_file(trip, person, config, logger):
    """Generate tours from Daysim-formatted trip records by iterating through person-days."""    

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

    # Build tour file; return df of tours and list of trips part of incomplete tours
    tour, trip = tours.create(trip, config)

    # After tour file is created, apply expression file for tours
    expr_df = pd.read_csv(os.path.join(config["input_dir"], "tour_expr.csv"))

    tour = convert.process_expression_file(tour, expr_df)

    # Assign weight toexpfac as hhexpfac (getting it from psexpfac, which is the same as hhexpfac)
    tour = tour.merge(person[["person_id", "psexpfac"]], on="person_id", how="left")
    tour.rename(columns={"psexpfac": "toexpfac"}, inplace=True)

    # # remove the trips that weren't included in the tour file
    # _filter = -trip["trip_id"].isin(bad_trips)
    # logger.info(f"Dropped {len(trip[~_filter])} total trips due to tour issues ")
    # trip = trip[_filter]
    # pd.DataFrame(bad_trips).T.to_csv(
    #     os.path.join(config["output_dir"], "bad_trips.csv")
    # )

    return tour, trip


def process_household_day(person_day_original_df, hh, config):
    person_day_original_df["household_id"] = person_day_original_df[
        "household_id"
    ].astype("int64")
    household_day = (
        person_day_original_df.groupby(["household_id", "travel_dow"])
        .count()
        .reset_index()[["household_id", "travel_dow"]]
    )

    # household_day.rename(
    #     columns={"household_id": "hhno", "travel_dow": "day"}, inplace=True
    # )

    # add day of week lookup
    household_day["dow"] = household_day["travel_dow"].copy()
    household_day["day"] = household_day["travel_dow"].copy()

    # Set number of joint tours to 0 for this version of Daysim
    for col in ["jttours", "phtours", "fhtours"]:
        household_day[col] = 0

    # Add an ID column
    household_day["id"] = range(1, len(household_day) + 1)

    # Add expansion factor
    household_day = household_day.merge(
        hh[["hhno", "hhexpfac"]],
        left_on="household_id",
        right_on="hhno",
        how="left",
    )
    household_day.rename(columns={'hhexpfac': "hdexpfac"}, inplace=True)
    household_day["hdexpfac"] = household_day["hdexpfac"].fillna(-1)

    return household_day

def process_person_day(
    tour, person, trip, hh, person_day_original_df, household_day, config
):
    # Get the usual workplace column from person records
    tour = tour.merge(person[["hhno", "pno", "pwpcl"]], on=["hhno", "pno"], how="left")

    # Build person day file directly from tours and trips
    pday = days.create(tour, person, trip, config)

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
    pday = pday.merge(
        person_day_original_df[["household_id", "pernum", "travel_dow", "wkathome"]],
        left_on=["hhno", "pno", "day"],
        right_on=["household_id", "pernum", "travel_dow"],
        how="inner",
    )

    no_travel_df = person_day_original_df[
        (person_day_original_df["num_trips"] == 0)
        & person_day_original_df["travel_dow_label"].isin(
            ["Monday", "Tuesday", "Wednesday", "Thursday"]
        )
    ]
    no_travel_df = no_travel_df[no_travel_df["person_id"].isin(person["person_id"])]
    # Only add entries for completed survey days
    # no_travel_df = no_travel_df[no_travel_df["svy_complete"] == "Complete"]

    pday["no_travel_flag"] = 0

    for person_rec in no_travel_df.person_id.unique():
        pday.loc[person_rec, :] = 0
        pday.loc[person_rec, "no_travel_flag"] = 1
        pday.loc[person_rec, "hhno"] = no_travel_df[
            no_travel_df["person_id"] == person_rec
        ]["household_id"].values[0]
        pday.loc[person_rec, "pno"] = no_travel_df[
            no_travel_df["person_id"] == person_rec
        ]["pernum"].values[0]
        pday.loc[person_rec, "person_id"] = no_travel_df[
            no_travel_df["person_id"] == person_rec
        ]["person_id"].values[0]
        pday.loc[person_rec, "person_id"] = person_rec
        # these will all be replaced after this step so set to 1
        pday.loc[person_rec, "day"] = 1
        pday.loc[person_rec, "beghom"] = 1
        pday.loc[person_rec, "endhom"] = 1
        pday.loc[person_rec, "wkathome"] = no_travel_df[
            no_travel_df["person_id"] == person_rec
        ]["wkathome"].values[0]
        pday.loc[person_rec, "pdexpfac"] = person[person["person_id"] == person_rec][
            "psexpfac"
        ].values[0]

    # Join household day ID to person Day
    pday = pday.merge(
        household_day[["hhno", "day", "id"]], on=["hhno", "day"], how="left"
    )
    pday.rename(columns={"id": "household_day_id"}, inplace=True)

    return pday

def convert_format(config):
    # Start log file
    logger = logcontroller.setup_custom_logger("convert_format_logger.txt", config)
    logger.info("--------------------convert_format.py STARTING--------------------")
    start_time = datetime.datetime.now()

    # Load geolocated survey data
    trip_original_df = pd.read_csv(
        os.path.join(config["output_dir"], "geolocated_trip.csv")
    )
    hh_original_df = pd.read_csv(
        os.path.join(config["output_dir"], "geolocated_hh.csv")
    )
    person_original_df = pd.read_csv(
        os.path.join(config["output_dir"], "geolocated_person.csv")
    )

    # Load expression and variable recoding files
    df_lookup = pd.read_csv(os.path.join(config["input_dir"], "data_lookup.csv"))
    person_expr_df = pd.read_csv(
        os.path.join(config["input_dir"], "person_expr.csv"),
        delimiter=",(?![^\(]*[\)])",
    )  # Exclude quotes within data
    hh_expr_df = pd.read_csv(os.path.join(config["input_dir"], "hh_expr.csv"))

    # Load Person Day data from Elmer
    person_day_original_df = util.load_elmer_table(config["elmer_person_day_table"])
    # Join travel day value
    _df = df_lookup[df_lookup["elmer_name"] == "travel_dow"]
    person_day_original_df = person_day_original_df.merge(
        _df[["elmer_value", "model_value"]],
        left_on="travel_dow",
        right_on="elmer_value",
        how="left",
    )
    person_day_original_df.rename(
        columns={"travel_dow": "travel_dow_label", "model_value": "travel_dow"},
        inplace=True,
    )

    # Get day of week from day ID
    trip_original_df.rename(columns={'travel_dow': 'travel_dow_label'}, inplace=True)
    trip_original_df = trip_original_df.merge(
        person_day_original_df[["day_id", "travel_dow"]],
        how="left",
        on="day_id",
    )

    # Create new Household and Person records for each travel day.
    # For trip/tour models we use this data as if each person-day were independent for multiple-day diaries
    (
        hh_original_df,
        trip_original_df,
        person_original_df,
        person_day_original_df,
    ) = convert.create_multiday_records(
        hh_original_df,
        trip_original_df,
        person_original_df,
        person_day_original_df,
        "household_id",
        config,
    )

    if config["debug"]:
        person_original_df = person_original_df.iloc[0:100]
        trip_original_df = trip_original_df[
            trip_original_df["person_id"].isin(person_original_df.person_id)
        ]
        trip_original_df = trip_original_df[
            trip_original_df["household_id"].isin(person_original_df.household_id)
        ]
        hh_original_df = hh_original_df[
            hh_original_df["household_id"].isin(person_original_df.household_id)
        ]
        person_day_original_df = person_day_original_df[
            person_day_original_df["household_id"].isin(person_original_df.household_id)
        ]

    # Recode person, household, and trip data
    person = convert.process_expression_file(
        person_original_df,
        person_expr_df,
        df_lookup[df_lookup["table"] == "person"],
    )

    hh = process_household_file(hh_original_df, person, df_lookup, config, logger)
    trip = process_trip_file(
        trip_original_df, person, person_day_original_df, df_lookup, config, logger
    )

    # Paid parking data is not available from Elmer; calculate from workplace location
    parcel_df = pd.read_csv(
        config["parcel_file_dir"],
        sep='\s+',
        usecols=["parcelid", "parkhr_p"],
    )
    person = person.merge(parcel_df, left_on="pwpcl", right_on="parcelid", how="left")
    # If person works at a parcel with paid parking, assume they pay to park
    # FIXME! come up with a better process for this...
    person.loc[person["parkhr_p"] > 0, "ppaidprk"] = 1
    # Note that the 2023 survey does not include data about employer-paid benefits
    # We may have to estimate using older data...


    # Make sure trips are properly ordered, where deptm is increasing for each person's travel day
    # Note that we also have some trips that continue into the following day
    # The "day" column stays the same but the depart/arrive date may change. 
    trip["person_id_int"] = trip["person_id"].astype("int64")
    trip = trip.sort_values(["person_id_int", "day", "depart_date", "deptm"])
    trip = trip.reset_index()

    # Create tour file and update the trip file with tour info
    tour, trip = build_tour_file(trip, person, config, logger)

    household_day = process_household_day(
        person_day_original_df, hh, config
    )

    # person day
    person_day = process_person_day(
        tour, person, trip, hh, person_day_original_df, household_day, config
    )

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
