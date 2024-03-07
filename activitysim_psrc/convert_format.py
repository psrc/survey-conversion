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

# from activitysim_psrc import infer
from activitysim.abm.models.util import canonical_ids as cid

pd.options.mode.chained_assignment = None  # default='warn'


def process_household_file(hh, person, df_lookup, config, logger):
    """Convert household columns and generate data from person table"""

    expr_df = pd.read_csv(os.path.join(config["input_dir"], "hh_expr.csv"))

    person_type_dict = config["person_type_dict"]

    # For each household, calculate total number of people in each person_type
    person_type_field = "ptype"
    hhid_col = "household_id"
    for person_type in person[person_type_field].unique():
        df = person[person["ptype"] == person_type]
        df = df.groupby("household_id")["household_id"].count()
        df.name = person_type_dict[str(int(person_type))]

        # Join to households
        hh = pd.merge(hh, df, how="left", left_on="household_id", right_index=True)
        hh[person_type_dict[str(int(person_type))]].fillna(0, inplace=True)
        hh[person_type_dict[str(int(person_type))]] = hh[
            person_type_dict[str(int(person_type))]
        ].astype("int")

    # Apply expression file input
    hh = convert.process_expression_file(
        hh, expr_df, config["hh_columns"], df_lookup[df_lookup["table"] == "household"]
    )

    # Remove households without parcels
    _filter = -hh["home_zone_id"].isnull()
    logger.info(f"Dropped {len(hh[~_filter])} households: missing parcels")
    hh = hh[_filter]

    return hh


def process_trip_file(df, person, day, df_lookup, config, logger):
    """Convert trip records to Daysim format."""

    # Trips to/from work are considered "usual workplace" only if dtaz == workplace TAZ
    # must join person records to get usual work and school location
    df = df.merge(
        person[["person_id", "PNUM", "workplace_zone_id", "school_zone_id"]],
        how="left",
        on="person_id",
    )

    # Calculate fields using an expression file
    expr_df = pd.read_csv(os.path.join(config["input_dir"], "trip_expr.csv"))

    # Start and end time
    df["arrtm"] = df["arrival_time_hour"] * 60 + df["arrival_time_minute"]
    df["deptm"] = df["depart_time_hour"] * 60 + df["depart_time_minute"]

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

    # Get transit submode
    # convert.assign_tour_mode(df, tour_dict, tour_id, config)
    df["access_mode"] = "WALK"
    df.loc[
        df["mode_acc"].isin(config["drive_to_transit_access_list"]), "access_mode"
    ] = "DRIVE"

    for col in ["mode_1", "mode_2", "mode_3", "mode_4"]:
        for mode, mode_list in config["submode_dict"].items():
            df.loc[df[col].isin(mode_list), mode + "_flag"] = 1

    # Use mode heirarchy to deterimine primary transit submode
    df["transit_mode"] = np.nan
    for mode in config["transit_heirarchy"]:
        df.loc[
            (df["transit_mode"].isnull()) & (df[mode + "_flag"] == 1), "transit_mode"
        ] = mode

    # Join access method to complete transit submode
    df["trip_mode"] = df["access_mode"] + "_" + df["transit_mode"]

    # We don't separate drive access by submode so reset those values
    df.loc[df["access_mode"] == "DRIVE", "trip_mode"] = config["TRANSIT_DRIVE_MODE"]

    df = convert.process_expression_file(
        df, expr_df, config["trip_columns"], df_lookup[df_lookup["table"] == "trip"]
    )

    # Filter out trips that started before 0 minutes after midnight
    df = convert.apply_filter(
        df,
        "trips",
        df["depart"] >= 0,
        logger,
        "trips started before 0 minutes after midnight",
    )

    df = convert.apply_filter(
        df,
        "trips",
        (-df["trip_weight"].isnull()) & (df["trip_weight"] > 0),
        logger,
        "no or null weight",
    )

    for col in ["origin", "destination", "depart"]:
        df = convert.apply_filter(
            df, "trips", (df[col] >= 0), logger, "missing or unusable " + col
        )

    # if arrtm/deptm > 24*60, subtract that
    df = convert.apply_filter(
        df,
        "trips",
        ~(
            (df["origin"] == df["destination"])
            & (df["opurp"] == df["dpurp"])
            & (df["opurp"] == 1)
        ),
        logger,
        "intrazonal work-related trips",
    )

    # Filter null trips
    for col in ["trip_mode", "opurp", "dpurp", "origin", "destination"]:
        df = convert.apply_filter(
            df, "trips", -df[col].isnull(), logger, col + " is null"
        )

    return df


def build_tour_file(trip, person, config, logger):
    """Generate tours from Daysim-formatted trip records by iterating through person-days."""

    # Keep track of error types
    error_dict = {
        "first O and last D are not home": 0,
        "different number of tour starts and ends at home": 0,
        "dpurp of previous trip does not match opurp of next trip": 0,
        "activity type of previous trip does not match next trip": 0,
        "different number of tour starts/ends at home": 0,
        "no trips in set": 0,
        "no purposes provided except change mode": 0,
    }

    trip = convert.apply_filter(
        trip,
        "trips",
        -((trip["opurp"] == trip["dpurp"]) & (trip["opurp"] == "Home")),
        logger,
        "trips have same origin/destination of home",
    )

    # Reset the index
    trip = trip.reset_index()

    # Map to standard columns
    df_mapping = pd.read_csv(os.path.join(config["input_dir"], "mapping.csv"))

    trip = convert.map_to_class(trip, "trip", df_mapping, "original_to_script")

    # Build tour file; return df of tours and list of trips part of incomplete tours
    tour, bad_trips = tours.create(trip, error_dict, config)

    # Assign weight tour_weight as hhexpfac (getting it from person_weight, which is the same as hhexpfac)
    tour = tour.merge(
        person[["person_id", "person_weight", "household_id_elmer"]],
        on="person_id",
        how="left",
    )
    tour.rename(columns={"person_weight": "tour_weight"}, inplace=True)

    # remove the trips that weren't included in the tour file
    _filter = -trip["trip_id"].isin(bad_trips)
    logger.info(f"Dropped {len(trip[~_filter])} total trips due to tour issues ")
    trip = trip[_filter]
    pd.DataFrame(bad_trips).T.to_csv(
        os.path.join(config["output_dir"], "bad_trips.csv")
    )

    # Calculate tour category
    tour.loc[tour["parent"] > 0, "tour_category"] = "atwork"
    tour.loc[tour["tour_category"] != "atwork", "tour_category"] = "non_mandatory"
    tour.loc[
        (tour["pdpurp"].isin(["work", "school"])) & (tour["tour_category"] != "atwork"),
        "tour_category",
    ] = "mandatory"

    # stop_frequency- does not include primary stop
    tour["outbound_stops"] = tour["tripsh1"] - 1
    tour["inbound_stops"] = tour["tripsh2"] - 1
    tour["stop_frequency"] = (
        tour["outbound_stops"].astype("int").astype("str")
        + "out"
        + "_"
        + tour["inbound_stops"].astype("int").astype("str")
        + "in"
    )

    return tour, trip, error_dict


def process_household_day(person_day_original_df, hh, config):
    person_day_original_df["household_id"] = person_day_original_df[
        "household_id"
    ].astype("int64")
    household_day = (
        person_day_original_df.groupby(["household_id", "travel_dow"])
        .count()
        .reset_index()[["household_id", "travel_dow"]]
    )

    household_day.rename(
        columns={"household_id": "hhno", "travel_dow": "day"}, inplace=True
    )

    # add day of week lookup
    household_day["dow"] = household_day["day"]

    # Set number of joint tours to 0 for this version of Daysim
    for col in ["jttours", "phtours", "fhtours"]:
        household_day[col] = 0

    # Add an ID column
    household_day["id"] = range(1, len(household_day) + 1)

    # Add expansion factor
    # FIXME: no weights yet, replace with the weights column when available
    hh[config["hh_weight_col"]] = 1
    household_day = household_day.merge(
        hh[["household_id", config["hh_weight_col"]]],
        left_on="hhno",
        right_on="household_id",
        how="left",
    )
    household_day.rename(columns={config["hh_weight_col"]: "hdexpfac"}, inplace=True)
    household_day["hdexpfac"] = household_day["hdexpfac"].fillna(-1)

    return household_day

def build_joint_tours(tour, trip, person, config, logger):
    expr_df = pd.read_csv(os.path.join(config["input_dir"], "joint_tour_expr.csv"))

    tour = convert.process_expression_file(tour, expr_df, None)

    # After we've set begin and end hours, make sure all tour ends are after beginings
    _filter = tour["end"] >= tour["start"]
    logger.info(f"Dropped {len(tour[~_filter])} tours: tour end < tour start time")
    tour = tour[_filter]

    tour["tour_type"] = tour["pdpurp"].copy()

    # Enforce canonical tours - there cannot be more than 2 mandatory work tours
    # Flag mandatory vs non-mandatory to trips by purpose (and include joint non-mandatory trips)
    # Borrowing procedure from infer.py
    tour["mandatory_status"] = tour["tour_category"].copy()
    tour.loc[tour["mandatory_status"] == "joint", "mandatory_status"] = "non_mandatory"
    group_cols = ["person_id", "mandatory_status", "tour_type"]
    tour["tour_type_num"] = (
        tour.sort_values(by=group_cols).groupby(group_cols).cumcount() + 1
    )
    tour = tour.sort_values(
        ["person_id", "day", "tour_category", "tour_type", "tlvorig"]
    )

    possible_tours = cid.canonical_tours()
    possible_tours_count = len(possible_tours)
    tour_num_col = "tour_type_num"
    tour["tour_type_id"] = tour.tour_type + tour["tour_type_num"].map(str)
    tour.tour_type_id = tour.tour_type_id.replace(
        to_replace=possible_tours, value=list(range(possible_tours_count))
    )
    tour["loc_tour_id"] = tour.tour_type + tour[tour_num_col].map(str)

    # Non-numeric tour_type_id results are non-canonical and should be removed.
    # FIXME: For now just remove the offensive tours; is it okay to continue using this person's day records otherwise?
    filter = pd.to_numeric(tour["tour_type_id"], errors="coerce").notnull()

    # Keep track of the records we removed
    # tour[~filter].to_csv(os.path.join(output_dir,'temp','tours_removed_non_canoncial.csv'))
    tour = tour[filter]

    # At-work tour purposes are slightly different
    # eat, business, maint
    atwork_map = {
        "work": "business",
        "shopping": "maint",
        "othmaint": "maint",
        "othdiscr": "maint",
        "eatout": "eat",
        "social": "maint",
        "school": "business",
        "escort": "maint",
    }
    tour.loc[tour["tour_category"] == "atwork", "tour_type"] = tour["tour_type"].map(
        atwork_map
    )

    # Merge person number in household (PNUM) onto tour file
    # tour = tour.merge(person[['person_id','PNUM']], on='person_id', how='left')

    # Identify joint tours from tour df
    df = tour.groupby(["totaz", "tdtaz", "start", "end", "hhno"]).count()
    df = df.reset_index()

    # Each of these tours occur more than once in the data (assuming more than 1 person is on this same tour in the survey)
    joint_tour = 1
    joint_tour_dict = {}
    skip_tour = []
    for index, row in tour.iterrows():
        if row.tour_id not in skip_tour:
            # print(row.tour_id)
            filter = (
                (tour.day == row.day)
                & (tour.totaz == row.totaz)
                & (tour.tdtaz == row.tdtaz)
                & (tour.start == row.start)
                & (tour.end == row.end)
                & (tour.hhno == row.hhno)
            )
            # Get total number of participants (total number of matching tours) and assign a participant number
            participants = len(tour[filter])

            if len(tour[filter]) > 1:
                joint_tour_dict[joint_tour] = tour.loc[filter, "tour_id"].values
                tour.loc[filter, "joint_tour"] = joint_tour
                tour.loc[filter, "participant_num"] = tour["pno"]
                # Flag to skip this tour's joint companion
                for i in tour.loc[filter, "tour_id"].values:
                    skip_tour.append(i)
                joint_tour += 1

    tour["participant_num"] = tour["participant_num"].fillna(0).astype("int")
    # Use the joint_tour field to identify joint tour participants
    # Output should be a list of people on each tour; use the tour ID of participant_num == 1
    # joint_tour_list = tour[tour['joint_tour'].duplicated()]['joint_tour'].values
    # df = tour[((tour['joint_tour'].isin(joint_tour_list)) & (~tour['joint_tour'].isnull()))]
    joint_tours = tour[~tour["joint_tour"].isnull()]

    # Drop any tours that are for work, school, or escort
    # FIXME: should we change the purpose for some of these?
    # Escort trips are likely not coded properly
    # The tour type can be changed
    joint_tours = joint_tours[
        ~joint_tours["tour_type"].isin(["Work", "School", "Escort"])
    ]
    # joint_tour_list = joint_tours[joint_tours['joint_tour'].duplicated()]['joint_tour'].values
    joint_tour_list = joint_tours["joint_tour"].unique()

    # # See if there are any other tours we missed by comparing against the trip-level number of participants
    # # Get list of non-joint tours
    # test_list = []
    # for tour_id in tour.loc[(tour['joint_tour'].isnull()) &
    #                         (~tour['tour_type'].isin(['work','school','escort'])), 'tour_id']:
    #     print(tour)
    #     _df = trip.loc[trip['tour_id'] == tour_id]
    #     if _df['travelers_hh'].sum() > len(_df)*2:
    #         # Do some further testing
    #         test_list.append(tour)

    # Assume Tour ID of first participant, so sort by joint_tour and person ID
    joint_tours = joint_tours.sort_values(["joint_tour", "person_id"])
    tour = tour.sort_values(["joint_tour", "person_id"])
    for joint_tour in joint_tour_list:
        joint_tours.loc[
            joint_tours["joint_tour"] == joint_tour, "tour_id"
        ] = joint_tours[joint_tours["joint_tour"] == joint_tour].iloc[0]["tour_id"]
        # Remove other tours except the primary tour from tour file completely;
        # These will only be accounted for in the joint_tour_file
        tour = tour[
            ~tour["tour_id"].isin(
                tour[tour["joint_tour"] == joint_tour].iloc[1:]["tour_id"]
            )
        ]
        # Set this tour as joint category
        tour.loc[tour["joint_tour"] == joint_tour, "tour_category"] = "joint"

    # Define participant ID as tour ID + participant num
    joint_tours["participant_id"] = joint_tours["tour_id"].astype("str") + joint_tours[
        "participant_num"
    ].astype("int").astype("str")

    joint_tours = joint_tours[
        ["person_id", "tour_id", "hhno", "participant_num", "participant_id"]
    ]
    # joint_tours[SURVEY_TOUR_ID] = joint_tours['tour_id'].copy()
    # joint_tours.to_csv(os.path.join(output_dir, 'survey_joint_tour_participants.csv'), index=False)

    ## Filter to remove any joint work mandatory trips
    # FIXME: do not remove all trips, just those of the additional person and modify to be non-joint
    tour = tour[
        ~(
            (tour["tour_type"].isin(["school", "work", "escort"]))
            & (tour["tour_category"] == "joint")
        )
    ]

    # These must be added to trip after tour info is available
    trip["outbound"] = False
    trip.loc[trip["half"] == 1, "outbound"] = True

    trip["trip_num"] = trip["tseg"].copy()

    return joint_tours, trip, tour


def update_ids(df, df_person):
    df = df.merge(
        df_person[["household_id", "person_id", "person_id_original"]],
        left_on="person_id",
        right_on="person_id_original",
        how="left",
    )
    for col in ["person", "household"]:
        df.drop([col + "_id_x"], axis=1, inplace=True)
        df.rename(columns={col + "_id_y": col + "_id"}, inplace=True)
        df[col + "_id"] = df[col + "_id"].astype("int32")

    return df


def reset_ids(person, person_day, households, trip, config):
    # activitysim checks specifically for int32 types; however, converting 64-bit int to 32 can create data issues if IDs are too long
    # Create a new mapping for person ID and household ID
    person["person_id_original"] = person["person_id"].copy()
    person["person_id"] = range(1, len(person) + 1)

    households["household_id_original"] = households["household_id"].copy()
    households["household_id"] = range(1, len(households) + 1)
    households["household_id"].astype("int32")

    # Merge new household ID to person records
    person = person.merge(
        households[["household_id", "household_id_original"]],
        left_on="household_id",
        right_on="household_id_original",
        how="left",
    )
    person.drop(["household_id_x"], axis=1, inplace=True)
    person.rename(columns={"household_id_y": "household_id"}, inplace=True)

    # Merge new household and person records to person day
    person_day["person_id"] = person_day["person_id"].astype("int64")

    person_day = update_ids(person_day, person)
    trip = update_ids(trip, person)

    person[
        ["person_id", "person_id_original", "household_id", "household_id_original"]
    ].to_csv(os.path.join(config["output_dir"], "person_and_household_id_mapping.csv"))

    return person, person_day, households, trip


def convert_format(config):
    # Start log file
    logger = logcontroller.setup_custom_logger("convert_format_logger.txt", config)
    logger.info("--------------------convert_format.py STARTING--------------------")
    start_time = datetime.datetime.now()

    # Load expression and variable recoding files
    df_lookup = pd.read_csv(os.path.join(config["input_dir"], "data_lookup.csv"))
    person_expr_df = pd.read_csv(
        os.path.join(config["input_dir"], "person_expr.csv"),
        delimiter=",(?![^\(]*[\)])",
    )  # Exclude quotes within data
    hh_expr_df = pd.read_csv(os.path.join(config["input_dir"], "hh_expr.csv"))

    # Load Person Day data from Elmer
    person_day_original_df = util.load_elmer_table(config["elmer_person_day_table"])
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

    # Get travel day label from person Day file
    trip_original_df.rename(columns={"travel_dow": "travel_dow_label"}, inplace=True)
    trip_original_df = trip_original_df.merge(
        person_day_original_df[["day_id", "travel_dow"]],
        how="left",
        on="day_id",
    )

    person, person_day, households, trip = reset_ids(
        person_original_df,
        person_day_original_df,
        hh_original_df,
        trip_original_df,
        config,
    )

    # Create new Household and Person records for each travel day.
    # For trip/tour models we use this data as if each person-day were independent for multiple-day diaries
    (
        households,
        trip,
        person,
        person_day,
    ) = convert.create_multiday_records(
        households,
        trip,
        person,
        person_day,
        "household_id",
        config,
    )

    # FIXME: it's hard to tie the original household/person ID back after reseting indexes and creating multiple days
    # Make sure an original hh_id flows through with the files

    # Recode person, household, and trip data
    person = convert.process_expression_file(
        person,
        person_expr_df,
        config["person_columns"],
        df_lookup[df_lookup["table"] == "person"],
    )

    hh = process_household_file(households, person, df_lookup, config, logger)
    trip = process_trip_file(trip, person, person_day, df_lookup, config, logger)

    # Make sure trips are properly ordered, where deptm is increasing for each person's travel day
    trip["person_id_int"] = trip["person_id"].astype("int64")
    trip = trip.sort_values(["person_id_int", "day", "depart"])
    trip = trip.reset_index()

    # Create tour file and update the trip file with tour info
    tour, trip, error_dict = build_tour_file(trip, person, config, logger)

    #
    joint_tour, trip, tour = build_joint_tours(tour, trip, person, config, logger)

    error_dict_df = pd.DataFrame(
        error_dict.values(), index=error_dict.keys(), columns=["errors"]
    )
    for index, row in error_dict_df.iterrows():
        logger.info(f"Dropped {row.errors} tours: " + index)

    # FIXME!!
    # Excercise trips are coded as -88, make sure those are excluded (?)

    # Set all travel days to 1
    trip["day"] = 1
    tour["day"] = 1

    # Activitysim significant clean up at this point

    trip[["travdist", "travcost", "travtime"]] = "-1.00"

    for df_name, df in {
        "survey_persons.csv": person,
        "survey_trips.csv": trip,
        "survey_tours.csv": tour,
        "survey_households.csv": hh,
        "survey_joint_tour_participants.csv": joint_tour,
    }.items():
        print(df_name)

        df.to_csv(
            os.path.join(config["output_dir"], df_name),
            index=False,
        )

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------convert_format.py ENDING--------------------")
    logger.info("convert_format.py RUN TIME %s" % str(elapsed_total))
