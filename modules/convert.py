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

pd.options.mode.chained_assignment = None  # default='warn'


def map_to_class(df_trip, df_mapping):
    """Create a dictionary to map between model format and this script."""
    df_mapping = df_mapping[~df_mapping["model_variable"].isnull()]
    df_mapping = df_mapping.set_index("model_variable")
    map_dict = df_mapping.to_dict()["class_variable"]

    df_trip.rename(columns=map_dict, inplace=True)

    return df_trip


def unique_person_id(df, hhid_col, pno_col):
    """Create an unique person ID from household and person number.
    Returns str
    """
    df["unique_person_id"] = df[hhid_col].astype("int64").astype("str") + df[
        pno_col
    ].astype("int64").astype("str")

    return df


def unique_household_id(df, hhid_col, day_col):
    """Create an unique person ID from household and person number.
    Returns int64
    """
    df["household_id"] = df[hhid_col].astype("int64").astype("str") + df[
        day_col
    ].astype("int64").astype("str")

    df["household_id"] = df["household_id"].astype("int64")

    return df


def apply_filter(df, df_name, filter, logger, msg):
    """Apply a filter to trim df, record changes in log file."""

    logger.info(f"Dropped {len(df[~filter])} " + df_name + ": " + msg)

    return df[filter]


def assign_tour_mode(_df, tour_dict, tour_id, config):
    """Get a list of transit modes and identify primary mode
    Primary mode is the first one from a heirarchy list found in the tour.
    Assumes modes are values and not strings.
    """
    mode_list = _df["mode"].value_counts().index.astype("int").values

    for mode in config["mode_heirarchy"]:
        if mode in mode_list:
            # If transit, check whether access mode is walk to transit or drive to transit
            if mode == config["walk_to_transit_mode"]:
                # Try to use the access mode field values to get access mode
                # FIXME: make this more general, note requirements
                if len(
                    [
                        i
                        for i in _df["mode_acc"].values
                        if i in config["drive_to_transit_access_list"]
                    ]
                ):
                    mode = config["drive_to_transit_mode"]  # park and ride

            return mode


def process_expression_file(df, expr_df, output_column_list, df_lookup=None):
    """Execute each row of calculations in an expression file.
    Apply mapping from CSV files
    Fill empty columns in output_column_list with -1."""

    # Apply direct mapping from lookup CSV
    if df_lookup is not None:
        for col in df_lookup["elmer_name"].unique():
            _df_lookup = df_lookup[(df_lookup["elmer_name"] == col)]
            model_var_name = _df_lookup["model_name"].iloc[0]
            df = df.merge(
                _df_lookup[["elmer_value", "model_value"]],
                left_on=col,
                right_on="elmer_value",
                how="left",
            )
            # Update column names
            if col == model_var_name:
                df.drop(col, inplace=True, axis=1)  # avoid duplicate cols
            df.drop("elmer_value", axis=1, inplace=True)
            df.rename(columns={"model_value": model_var_name}, inplace=True)

    for index, row in expr_df.iterrows():
        expr = (
            "df.loc["
            + row["filter"]
            + ', "'
            + row["result_col"]
            + '"] = '
            + str(row["result_value"])
        )

        exec(expr)
    # Add empty columns to fill in later with skims
    for col in output_column_list:
        if col not in df.columns:
            df[col] = -1
        else:
            df[col] = df[col].fillna(-1)

    df = df[output_column_list]

    return df


def get_primary_index(df):
    if len(df) == 2:
        primary_index = df.index[0]
    else:
        # Identify the primary purpose
        primary_index = df[-df["dpurp"].isin([0, 10])]["duration"].idxmax()

    return primary_index


def add_tour_data(df, tour_dict, tour_id, day, config, primary_index=None):
    """
    Add tour data that is standard for all trip sets. This includes
    household and person data that is always available on the first trip record.
    Tour departure is always the departure time from the first trip.
    """

    for col in ["hhno", "pno", "person_id", "unique_person_id"]:
        tour_dict[tour_id][col] = df.iloc[0][col]

    tour_dict[tour_id]["day"] = day
    tour_dict[tour_id]["tour"] = tour_id

    # First trip row contains departure time and origin info
    tour_dict[tour_id]["tlvorig"] = df.iloc[0]["deptm"]
    tour_dict[tour_id]["totaz"] = df.iloc[0]["otaz"]
    if "opcl" in df.columns:
        tour_dict[tour_id]["topcl"] = df.iloc[0]["opcl"]
    tour_dict[tour_id]["toadtyp"] = df.iloc[0]["oadtyp"]

    # Last trip row contains return info
    tour_dict[tour_id]["tarorig"] = df.iloc[-1]["arrtm"]

    # If only 2 trips in tour, primary purpose is first trip's purpose
    if not primary_index:
        primary_index = get_primary_index(df)

    tour_dict[tour_id]["pdpurp"] = df.loc[primary_index]["dpurp"]
    tour_dict[tour_id]["tlvdest"] = df.loc[primary_index]["deptm"]
    tour_dict[tour_id]["tdtaz"] = df.loc[primary_index]["dtaz"]
    if "dpcl" in df.columns:
        tour_dict[tour_id]["tdpcl"] = df.loc[primary_index]["dpcl"]
    tour_dict[tour_id]["tdadtyp"] = df.loc[primary_index]["dadtyp"]
    tour_dict[tour_id]["tardest"] = df.iloc[-1]["arrtm"]
    tour_dict[tour_id]["tripsh1"] = len(df.loc[0:primary_index])
    tour_dict[tour_id]["tripsh2"] = len(df.loc[primary_index + 1 :])
    tour_dict[tour_id]["tmodetp"] = assign_tour_mode(df, tour_dict, tour_id, config)

    # path type
    # Pathtype is defined by a heirarchy, where highest number is chosen first
    # Ferry > Commuter rail > Light Rail > Bus > Auto Network
    # Note that tour pathtype is different from trip path type (?)
    if "pathtype" in df.columns:
        tour_dict[tour_id]["tpathtp"] = df.loc[df["mode"].idxmax()]["pathtype"]

    return tour_dict


def update_trip_data(trip, df, tour_id):
    """
    trip: dataframe of trips to be updated
    df: dataframe of selected tt
    """

    primary_index = get_primary_index(df)

    trip.loc[trip["trip_id"].isin(df["trip_id"].values), "tour"] = tour_id

    trip.loc[
        trip["trip_id"].isin(df.loc[0:primary_index].trip_id),
        "half",
    ] = 1
    trip.loc[
        trip["trip_id"].isin(df.loc[primary_index + 1 :].trip_id),
        "half",
    ] = 2

    # set trip segment within half tours
    trip.loc[
        trip["trip_id"].isin(df.loc[0:primary_index].trip_id),
        "tseg",
    ] = range(1, len(df.loc[0:primary_index]) + 1)
    trip.loc[
        trip["trip_id"].isin(df.loc[primary_index + 1 :].trip_id),
        "tseg",
    ] = range(1, len(df.loc[primary_index + 1 :]) + 1)

    trip.loc[trip["trip_id"].isin(df["trip_id"].values), "tour"] = tour_id

    return trip


def create_multiday_records(hh, trip, person, person_day, hhid_col, config):
    """Treat different travel days as unique person/household records.
    Modify household ID as original ID concatenated with travel day.
    Replicate household and person records to match these "new" person/households.
    """

    hh["hhid_elmer"] = hh[hhid_col].copy()
    trip["hhid_elmer"] = trip[hhid_col].copy()
    person["hhid_elmer"] = person[hhid_col].copy()
    person_day["hhid_elmer"] = person_day[hhid_col].copy()

    person_day = unique_household_id(person_day, "hhid_elmer", "travel_dow")
    trip = unique_household_id(trip, "hhid_elmer", "travel_dow")

    multiday_hh = pd.DataFrame()
    multiday_person = pd.DataFrame()

    # Build person and household records for each of these person days
    for day in person_day["travel_dow"].unique():
        _hh = hh.loc[
            hh["hhid_elmer"].isin(
                person_day[person_day["travel_dow"] == day].hhid_elmer
            )
        ].copy()
        _hh["travel_dow"] = day
        _hh = unique_household_id(_hh, "hhid_elmer", "travel_dow")
        multiday_hh = multiday_hh.append(_hh)

        _person = person.loc[
            person["hhid_elmer"].isin(
                person_day[person_day["travel_dow"] == day].hhid_elmer
            )
        ].copy()
        _person["travel_dow"] = day
        _person = unique_household_id(_person, "hhid_elmer", "travel_dow")
        multiday_person = multiday_person.append(_person)

    multiday_person.drop("travel_dow", axis=1, inplace=True)
    multiday_hh.drop("travel_dow", axis=1, inplace=True)

    assert (
        len(
            multiday_person[
                ~multiday_person["household_id"].isin(multiday_hh.household_id)
            ]
        )
        == 0
    )
    assert len(trip[~trip["household_id"].isin(multiday_hh.household_id)]) == 0
    assert (
        len(person_day[~person_day["household_id"].isin(multiday_hh.household_id)]) == 0
    )

    # Generate unique person ID since the Elmer person_id has been copied for multiple days
    trip = unique_person_id(trip, "household_id", "pernum")
    multiday_person = unique_person_id(multiday_person, "household_id", "pernum")
    person_day = unique_person_id(person_day, "household_id", "pernum")

    # Write lookup between new IDs and original Elmer IDs
    # hh[['hhid_elmer',hhid_col]].to_csv(os.path.join(config['output_dir'], "household_hhno_mapping.csv"), index=False)

    return multiday_hh, trip, multiday_person, person_day
