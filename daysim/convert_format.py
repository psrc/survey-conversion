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
        hh, expr_df, config["hh_columns"], df_lookup[df_lookup["table"] == "household"]
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

    # Get day of week from day ID
    df = df.merge(day[["day_id", "travel_dow"]], how="left", on="day_id")

    # Calculate fields using an expression file
    expr_df = pd.read_csv(os.path.join(config["input_dir"], "trip_expr.csv"))

    # Select only weekday trips (M-Th)
    df = convert.apply_filter(
        df,
        "trips",
        df["travel_dow"].isin(["Monday", "Tuesday", "Wednesday", "Thursday"]),
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

    # Start and end time
    df["arrtm"] = df["arrival_time_hour"] * 60 + df["arrival_time_minute"]
    df["deptm"] = df["depart_time_hour"] * 60 + df["depart_time_minute"]

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
        df, expr_df, config["trip_columns"], df_lookup[df_lookup["table"] == "trip"]
    )

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

    # Build tour file; return df of tours and list of trips part of incomplete tours
    tour, bad_trips = tours.create(trip, error_dict, config)

    # After tour file is created, apply expression file for tours
    expr_df = pd.read_csv(os.path.join(config["input_dir"], "tour_expr.csv"))

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

    trip = trip[config["trip_columns"] + ["unique_person_id", "household_id_elmer"]]

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


def process_person_day(tour, person, trip, hh, person_day_original_df, config):
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
    pday["day"] = pday["day"].astype("int64")
    pday["person_id"] = pday["person_id"].astype("int64")
    person_day_original_df["day"] = person_day_original_df["day"].astype("int64")
    person_day_original_df["person_id"] = person_day_original_df["person_id"].astype(
        "int64"
    )
    pday["household_id_elmer"] = pday["household_id_elmer"].astype("int64")
    pday = pday.merge(
        person_day_original_df[["person_id", "day", "wkathome"]],
        left_on=["person_id", "day"],
        right_on=["person_id", "day"],
        how="inner",
    )

    # Add person day records of no travel
    # Get the unique person ID to merge with person records
    person_day_original_df = person_day_original_df.merge(
        person[["person_id", "hhno", "pno", "unique_person_id"]],
        on="person_id",
        how="left",
    )

    no_travel_df = person_day_original_df[person_day_original_df["num_trips"] == 0]
    no_travel_df = no_travel_df[
        no_travel_df["unique_person_id"].isin(person["unique_person_id"])
    ]

    # FIXME: for 2023 this is only available on trips now
    # # Only add entries for completed survey days
    # no_travel_df = no_travel_df[no_travel_df["svy_complete"] == "Complete"]

    pday["no_travel_flag"] = 0

    for person_rec in no_travel_df.unique_person_id.unique():
        pday.loc[person_rec, :] = 0
        pday.loc[person_rec, "no_travel_flag"] = 1
        pday.loc[person_rec, "hhno"] = no_travel_df[
            no_travel_df["unique_person_id"] == person_rec
        ]["hhno"].values[0]
        pday.loc[person_rec, "pno"] = no_travel_df[
            no_travel_df["unique_person_id"] == person_rec
        ]["pno"].values[0]
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
    person_day_original_df.rename(columns={"model_value": "day"}, inplace=True)
    person_day_original_df["day"] = person_day_original_df["day"].astype("int")

    if not config["debug_tours"]:
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

        # Recode person, household, and trip data
        person = convert.process_expression_file(
            person_original_df,
            person_expr_df,
            config["person_columns"],
            df_lookup[df_lookup["table"] == "person"],
        )

        hh = process_household_file(hh_original_df, person, df_lookup, config, logger)
        trip = process_trip_file(
            trip_original_df, person, person_day_original_df, df_lookup, config, logger
        )

        # Calculate person fields using trips with updated format

        # Write mapping between original trip_id and tsvid used on they survey
        trip[["trip_id", "tsvid"]].to_csv(
            os.path.join(config["output_dir"], "trip_id_tsvid_mapping.csv")
        )

        # Create new Household and Person records for each travel day.
        # For trip/tour models we use this data as if each person-day were independent for multiple-day diaries
        hh["household_id_elmer"] = hh["hhno"].copy()
        trip["household_id_elmer"] = trip["hhno"].copy()
        person["household_id_elmer"] = person["hhno"].copy()

        hh["new_hhno"] = hh["hhno"].copy()
        hh["flag"] = 0

        for day in trip["day"].unique():
            day = int(day)
            trip.loc[trip["day"] == day, "new_hhno"] = (
                trip["hhno"].astype("int") * 10 + day
            )
            trip["new_hhno"] = trip["new_hhno"].fillna(-1).astype("int64")

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
    ].astype("int64").astype("str")
    person["unique_person_id"] = person["hhno"].astype("int64").astype("str") + person[
        "pno"
    ].astype("int64").astype("str")

    # Create tour file and update the trip file with tour info
    tour, trip, error_dict = build_tour_file(trip, person, config, logger)
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
    person_day = process_person_day(
        tour, person, trip, hh, person_day_original_df, config
    )

    # FIXME!!
    # For person file, calculate whether parking was paid at the workplace

    # FIXME!!
    # Excercise trips are coded as -88, make sure those are excluded (?)

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
