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
import json
import datetime
import pandas as pd
import numpy as np
import h5py
import glob
import math
from shutil import copyfile
import logging
from daysim import logcontroller
import toml


def text_to_dictionary(input_filename):
    """Convert text input to Python dictionary."""
    my_file = open(input_filename)
    my_dictionary = {}

    for line in my_file:
        k, v = line.split(":")
        my_dictionary[eval(k)] = v.strip()

    return my_dictionary


def write_skims(df, skim_dict, otaz_field, dtaz_field, skim_output_file, config):
    """Look up skim values from trip records and export as csv."""

    dictZoneLookup = json.load(open(os.path.join(config['input_dir'], "zone_dict.txt")))
    dictZoneLookup = {int(k): int(v) for k, v in dictZoneLookup.items()}

    bikewalk_tod = "5to6"  # bike and walk are only assigned in 5to6
    distance_skim_tod = "7to8"  # distance skims don't change over time, only saved for a single time period

    output_array = []

    for i in range(len(df)):
        rowdata = df.iloc[i]
        rowresults = {}

        if rowdata["dephr"] == -1:
            next

        rowresults["id"] = rowdata["id"]
        rowresults["skimid"] = rowdata["skim_id"]
        rowresults["tod_orig"] = rowdata["dephr"]

        # write transit in vehicle times
        if rowdata["mode code"] == "ivtwa":
            tod = "7to8"
            rowresults["tod_pulled"] = tod

            try:
                my_matrix = skim_dict[tod]["Skims"]["ivtwa"]

                otaz = rowdata[otaz_field]
                dtaz = rowdata[dtaz_field]

                skim_value = my_matrix[dictZoneLookup[otaz]][dictZoneLookup[dtaz]]
                rowresults["t"] = skim_value

                my_matrix = skim_dict[tod]["Skims"]["sov_inc2d"]
                skim_value = my_matrix[dictZoneLookup[otaz]][dictZoneLookup[dtaz]]
                rowresults["d"] = skim_value

                # fare data is only available for 6to7 time period for AM peak, or 9to10 for mid-dat (off-peak)
                # assuming all trips are peak for now
                my_matrix = skim_dict["6to7"]["Skims"]["mfafarbx"]
                skim_value = my_matrix[dictZoneLookup[otaz]][dictZoneLookup[dtaz]]
                rowresults["c"] = skim_value

            # if value unavailable, keep going and assign -1 to the field
            except:
                rowresults["t"] = -1
                rowresults["d"] = skim_value
                rowresults["c"] = -1
        else:
            for skim_type in ["d", "t", "c"]:
                tod = rowdata["dephr"]

                # assign alternate tod value for special cases
                if skim_type == "d":
                    tod = distance_skim_tod

                if rowdata["mode code"] in ["bike", "walk"]:
                    tod = "5to6"

                rowresults["tod_pulled"] = tod

                # write results out
                try:
                    my_matrix = skim_dict[tod]["Skims"][rowdata["skim_id"] + skim_type]

                    otaz = rowdata[otaz_field]
                    dtaz = rowdata[dtaz_field]

                    skim_value = my_matrix[dictZoneLookup[otaz]][dictZoneLookup[dtaz]]
                    rowresults[skim_type] = skim_value
                # if value unavailable, keep going and assign -1 to the field
                except:
                    rowresults[skim_type] = -1

        output_array.append(rowresults)

    df = pd.DataFrame(output_array)
    # df.to_csv(os.path.join(config['output_dir'],'skims_attached','before_bike_edits.csv'), index=False)

    # For bike and walk skims, calculate distance from time skims using average speeds
    for mode, speed in {
        "bike": config["bike_speed"],
        "walk": config["walk_speed"],
    }.items():
        row_index = df["skimid"] == mode
        df.loc[row_index, "d"] = (df["t"] * speed / 60).astype("int")

    # Replace all bike and walk cost skims with 0
    df.loc[df["skimid"].isin(["bike", "walk"]), "c"] = 0
    # df.to_csv(
    #     os.path.join(config["output_dir"], "skims_attached", "initial", "after_bike_edits.csv"),
    #     index=False,
    # )

    # write results to a csv
    try:
        df.to_csv(
            os.path.join(config["output_dir"], "skims_attached", "initial", skim_output_file),
            index=False,
        )
    except:
        print("failed on export of output")


def fetch_skim(
    df_name, df, time_field, mode_field, otaz_field, dtaz_field, config, use_mode=False
):
    """
    Look up skim values form survey records.
    Survey fields required:
    Household income in dollars,
    time field (departure/arrival) in hhmm,
    optional: use standard mode for all skims (for auto distance/time skims only)

    """

    # Filter our records with missing data for required skim fields
    df = df[df[time_field] >= 0]
    df = df[df[otaz_field] >= 0]
    df = df[df[dtaz_field] >= 0]

    # Build a lookup variable to find skim value
    matrix_dict_loc = os.path.join(
        config["run_root"], r"inputs\model\skim_parameters\demand_matrix_dictionary.txt"
    )
    matrix_dict = text_to_dictionary(matrix_dict_loc)
    uniqueMatrices = set(matrix_dict.values())

    skim_output_file = df_name + "_skim_output.csv"

    # Use the same toll preference for all skims
    df["Toll Class"] = np.ones(len(df))

    # Convert continuous VOT to bins (0-15,15-25,25+) based on average household income
    # Note that all households with -1 (missing income) represent university students
    # These households are lumped into the lowest VOT bin 1,
    # FIXME: what is source of these income bins? Move to config!
    df["VOT Bin"] = pd.cut(
        df["income"],
        bins=[-1, 84500, 108000, 9999999999],
        right=True,
        labels=[1, 2, 3],
        retbins=False,
        precision=3,
        include_lowest=True,
    )

    df["VOT Bin"] = df["VOT Bin"].astype("int")
    df["dephr"] = np.floor(df[time_field]/60).astype('int').astype('str').map(config["tod_dict"])
    df[mode_field] = df[mode_field].fillna(-99)
    modes = np.asarray(df[mode_field].astype("str"))
    if use_mode:
        df["mode code"] = [config["skim_mode_dict"][use_mode] for i in range(len(df))]
    else:
        df["mode code"] = [config["skim_mode_dict"][modes[i]] for i in range(len(df))]

    # Concatenate to produce ID to use with skim tables
    # but not for walk or bike modes
    # mfivtwa

    final_df = pd.DataFrame()
    for mode in np.unique(df["mode code"]):
        mylen = len(df[df["mode code"] == mode])
        tempdf = df[df["mode code"] == mode]
        if mode not in ["walk", "bike", "ivtwa"]:
            tempdf["skim_id"] = (
                tempdf["mode code"] + "_inc" + tempdf["VOT Bin"].astype("str")
            )
        else:
            tempdf["skim_id"] = tempdf["mode code"]
        final_df = pd.concat([final_df,tempdf])
        print("number of " + mode + "trips: " + str(len(final_df)))
    df = final_df
    del final_df

    # Load skim data from h5 into a dictionary
    tods = set(config["tod_dict"].values())
    skim_dict = {}
    for tod in tods:
        contents = h5py.File(
            os.path.join(config["run_root"], r"inputs/model/roster", tod + ".h5")
        )
        skim_dict[tod] = contents
    
    # If the skim output file doesn't already exist, create it
    write_skims(df, skim_dict, otaz_field, dtaz_field, skim_output_file, config)


def process_person_skims(tour, person, hh, config):
    """
    Add person and HH level data to trip records.
    """

    tour_per = pd.merge(tour, person, on=["person_id"], how="left")
    tour_per["unique_id"] = (tour_per.person_id.astype("str"))
    tour_per["unique_tour_id"] = (
        tour_per["unique_id"]
        + "_"
        + tour_per["day"].astype("str")
        + "_"
        + tour_per["tour"].astype("str")
    )

    # Use tour file to get work departure/arrival time and mode
    work_tours = tour_per[tour_per["pdpurp"] == 'work']

    # Fill fields for usual work mode and times
    work_tours["puwmode"] = work_tours["tmodetp"]
    work_tours["puwarrp"] = work_tours["tlvorig"]
    work_tours["puwdepp"] = work_tours["tlvdest"]

    # some people make multiple work tours; select only the tours with greatest distance
    primary_work_tour = work_tours.groupby("unique_id")[[
        "tlvdest", "unique_tour_id"
    ]].max()
    work_tours = work_tours[
        work_tours.unique_tour_id.isin(primary_work_tour["unique_tour_id"].values)
    ]

    # drop the original Work Mode field
    # person.drop(["puwmode"], axis=1, inplace=True)

    person = pd.merge(
        person,
        work_tours[["person_id", "puwmode", "puwarrp", "puwdepp"]],
        on=["person_id"],
        how="left",
    )

    # Fill NA for this field with -1
    for field in ["puwmode", "puwarrp", "puwdepp"]:
        person[field].fillna(-1, inplace=True)

    # For people that didn't make a work tour but have a workplace location, use average departure time for skims
    median_puwarrp = person.loc[person['puwarrp'] > -1, 'puwarrp'].median()
    person.loc[(person['work_taz'] > 0) & (person['puwarrp'] == -1), 'puwarrp'] = median_puwarrp
    median_puwdepp = person.loc[person['puwdepp'] > -1, 'puwdepp'].median()
    person.loc[(person['work_taz'] > 0) & (person['puwdepp'] == -1), 'puwdepp'] = median_puwdepp


    # Get school tour info
    # pusarrp and pusdepp are non-daysim variables, meaning usual arrival and departure time from school
    school_tours = tour_per[tour_per["pdpurp"] == 'school']
    school_tours["pusmode"] = school_tours["tmodetp"]
    school_tours["pusarrp"] = school_tours["tlvorig"]
    school_tours["pusdepp"] = school_tours["tlvdest"]

    # Select a primary school trip, based on longest distance
    primary_school_tour = school_tours.groupby("unique_id")[[
        "tlvdest", "unique_tour_id"
    ]].max()
    school_tours = school_tours[
        school_tours["unique_tour_id"].isin(
            primary_school_tour["unique_tour_id"].values
        )
    ]

    person = pd.merge(
        person,
        school_tours[["person_id", "pusmode", "pusarrp", "pusdepp"]],
        on=["person_id"],
        how="left",
    )

    for field in ["pusarrp", "pusdepp"]:
        person[field].fillna(-1, inplace=True)

    # For people that didn't make a school tour but have a school location, use average departure time for skims
    median_pusarrp = person.loc[person['pusarrp'] > -1, 'pusarrp'].median()
    person.loc[(person['school_loc_taz'] > 0) & (person['pusarrp'] == -1), 'pusarrp'] = median_pusarrp
    median_pusdepp = person.loc[person['pusdepp'] > -1, 'pusdepp'].median()
    person.loc[(person['school_loc_taz'] > 0) & (person['pusdepp'] == -1), 'pusdepp'] = median_pusdepp

    # Attach income and TAZ info
    person = pd.merge(person, hh[["household_id", "income", "home_zone_id"]], on="household_id", how="left")

    # Fill -1 income (college students) with lowest income category
    min_income = person[person["income"] > 0]["income"].min()
    person.loc[person["income"] > 0, "income"] = min_income

    # Convert fields to int
    # for field in ["pusarrp", "pusdepp", "puwarrp", "puwdepp"]:
    #     person[field] = person[field].astype("int")

    # Write results to CSV for new derived fields
    person.to_csv(
        os.path.join(config["output_dir"], "skims_attached", "initial", "person_skim_output.csv"),
        index=False,
    )

    return person


def update_records(trip, tour, person, config):
    """
    Add skim value results to original survey files.
    """

    # Load skim data
    trip_skim = pd.read_csv(
        os.path.join(config["output_dir"], "skims_attached", "initial",  "trip_skim_output.csv")
    )
    tour_skim = pd.read_csv(
        os.path.join(config["output_dir"], "skims_attached", "initial", "tour_skim_output.csv")
    )
    person_skim = pd.read_csv(
        os.path.join(config["output_dir"], "skims_attached", "initial", "person_skim_output.csv")
    )
    work_skim = pd.read_csv(
        os.path.join(
            config["output_dir"], "skims_attached", "initial", "work_travel_skim_output.csv"
        )
    )
    school_skim = pd.read_csv(
        os.path.join(
            config["output_dir"], "skims_attached", "initial", "school_travel_skim_output.csv"
        )
    )

    for df in [trip_skim, tour_skim, person_skim, work_skim, school_skim]:
        df["id"] = df["id"].astype("int")

    for df in [trip, tour, person]:
        df["id"] = df["id"].astype("int")

    trip_cols = {"trip_time": "t", "trip_cost": "c", "trip_distance": "d"}
    tour_cols = {"tour_time": "t", "tour_cost": "c", "tour_distance": "d"}
    person_cols = {"puwmode": "puwmode"}
    work_cols = {"time_to_work": "t", "distance_to_work": "d"}
    school_cols = {"time_to_school": "t", "distance_to_school": "d"}


    # drop skim columns from the old file
    # trip.drop(trip_cols.keys(), axis=1, inplace=True)
    # # tour.drop(tour_cols.keys(), axis=1, inplace=True)
    # person.drop(person_cols.keys(), axis=1, inplace=True)
    # person.drop(work_cols.keys(), axis=1, inplace=True)
    # person.drop(school_cols.keys(), axis=1, inplace=True)

    # Join skim file to original
    df = pd.merge(trip, trip_skim[["id", "c", "d", "t"]], on="id", how="left")
    for colname, skimname in trip_cols.items():
        df[colname] = df[skimname]
        df.drop(skimname, axis=1, inplace=True)

        # divide skims by 100
        df[colname] = (
            df[df[colname] >= 0][colname].iloc[:] / 100
        )  # divide all existing skim values by 100
        df[colname].fillna(-1.0, inplace=True)
    df = df.drop(["id"], axis=1)

    # export results
    df.to_csv(
        os.path.join(config["output_dir"], "skims_attached", "initial", "survey_trips.csv"),
        index=False,
    )

    # For tour
    df = pd.merge(tour, tour_skim[["id", "c", "d", "t"]], on="id", how="left")
    for colname, skimname in tour_cols.items():
        df[colname] = df[skimname]
        df.drop(skimname, axis=1, inplace=True)
        df[colname] = (
            df[df[colname] >= 0][colname].iloc[:] / 100
        )  # divide all existing skim values by 100
        df[colname].fillna(-1.0, inplace=True)
    df = df.drop(["id"], axis=1)

    # export results
    df.to_csv(
        os.path.join(config["output_dir"], "skims_attached", "initial", "survey_tours.csv"),
        index=False,
    )

    # Person records
    df = pd.merge(
        person,
        person_skim[["id", "puwmode", "puwarrp", "puwdepp"]],
        on="id",
        how="left",
    )

    df = pd.merge(df, work_skim[["id", "d", "t"]], on="id", how="left")
    for colname, skimname in work_cols.items():
        df[colname] = df[skimname]
        df.drop(skimname, axis=1, inplace=True)
        df[colname] = (
            df[df[colname] >= 0][colname].iloc[:] / 100
        )  # divide skim values by 100
        df[colname].fillna(-1.0, inplace=True)

    df = pd.merge(df, school_skim[["id", "d", "t"]], on="id", how="left")
    for colname, skimname in school_cols.items():
        df[colname] = df[skimname]
        df.drop(skimname, axis=1, inplace=True)
        df[colname] = (
            df[df[colname] >= 0][colname].iloc[:] / 100
        )  # divide skim values by 100
        df[colname].fillna(-1.0, inplace=True)

    df = df.drop(["id"], axis=1)

    # export results
    df.to_csv(
        os.path.join(config["output_dir"],"skims_attached", "initial", "survey_persons.csv"),
        index=False,
    )


def attach_skims(config, state):
    # Start log file
    logger = logcontroller.setup_custom_logger("attach_skims_logger.txt", config)
    logger.info("--------------------attach_skims.py STARTED--------------------")
    start_time = datetime.datetime.now()

    # Create new output directory for survey records with skims attached, if needed
    from pathlib import Path
    if not os.path.exists(os.path.join(config["output_dir"], "skims_attached", "initial/")):
        os.makedirs(os.path.join(config["output_dir"], "skims_attached", "initial"))

    # Load daysim-converted data produced from daysim_conversion.py
    input_dir = os.path.join(config["output_dir"])
    trip = pd.read_csv(os.path.join(input_dir, "survey_trips.csv"))
    tour = pd.read_csv(os.path.join(input_dir, "survey_tours.csv"))
    hh = pd.read_csv(os.path.join(input_dir, "survey_households.csv"))
    person = pd.read_csv(os.path.join(input_dir, "survey_persons.csv"))

    # Join household to trip data to get income
    trip_hh = pd.merge(trip, person, on="person_id")
    trip_hh = pd.merge(trip_hh, hh, on='household_id')
    tour_hh = pd.merge(tour, hh, left_on='hhno', right_on="household_id")

    # Add unique id fields
    person["id"] = person["person_id"].astype("int")
    trip_hh["id"] = trip_hh["tsvid"].astype("int")
    tour_hh["id"] = tour["tour_id"].astype("int")
    tour["id"] = tour["tour_id"].astype("int")
    trip["id"] = trip["tsvid"].astype("int")

    # Extract person-level results from trip file
    person_modified = process_person_skims(tour, person, hh, config)

    # Fetch trip skims based on trip departure time
    fetch_skim(
        "trip",
        trip_hh,
        time_field="deptm",   # minutes after midnight format
        mode_field="mode",
        otaz_field="otaz",
        dtaz_field="dtaz",
        config=config,
    )

    # Fetch tour skims based on tour departure time from origin
    fetch_skim(
        "tour",
        tour_hh,
        time_field="tlvorig",    # minutes after midnight format
        mode_field="tmodetp",
        otaz_field="totaz",
        dtaz_field="tdtaz",
        config=config,
    )

    # Attach person-level work skims based on home to work auto trips
    fetch_skim(
        "work_travel",
        person_modified,
        time_field="puwarrp",
        mode_field="puwmode",
        otaz_field="home_taz",
        dtaz_field="work_taz",
        config=config,
        use_mode="DRIVEALONEFREE",
    )

    # Attach person-level school skims based on home to school auto trips
    fetch_skim(
        "school_travel",
        person_modified,
        time_field="pusarrp",
        mode_field="pusmode",
        otaz_field="home_taz",
        dtaz_field="school_taz",
        config=config,
        use_mode="DRIVEALONEFREE",
    )

    # Reload original person file and attach skim results
    person = pd.read_csv(os.path.join(config["output_dir"], "survey_persons.csv"))
    person["id"] = person["person_id"].astype("int")

    # Update records
    update_records(trip, tour, person, config)

    # Write results to h5
    write_list = ["survey_households","survey_joint_tour_participants"]

    # Copy non-updated files to the same output directory for consistency
    for file in write_list:
        df = pd.read_csv(
            os.path.join(config["output_dir"], file + ".csv")
        )
        df.to_csv(
            os.path.join(config["output_dir"], "skims_attached", "initial", file + ".csv"),
            index=False,
        )

    # dat_to_h5(
    #     [
    #         os.path.join(config["output_dir"], "skims_attached", "initial", "_" + file + ".tsv")
    #         for file in write_list
    #     ],
    #     config,
    # )
