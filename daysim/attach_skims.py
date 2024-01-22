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
import logcontroller
import toml

config = toml.load("configuration.toml")

# Start log file
logger = logcontroller.setup_custom_logger("attach_skims_logger.txt")
logger.info("--------------------attach_skims.py STARTED--------------------")
start_time = datetime.datetime.now()


def text_to_dictionary(input_filename):
    """Convert text input to Python dictionary."""
    my_file = open(input_filename)
    my_dictionary = {}

    for line in my_file:
        k, v = line.split(":")
        my_dictionary[eval(k)] = v.strip()

    return my_dictionary


def write_skims(df, skim_dict, otaz_field, dtaz_field, skim_output_file):
    """Look up skim values from trip records and export as csv."""

    dictZoneLookup = json.load(open(os.path.join("inputs", "zone_dict.txt")))
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

                # assign atlernate tod value for special cases
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
    df.to_csv(
        os.path.join(config["output_dir"], "skims_attached", "after_bike_edits.csv"),
        index=False,
    )

    # write results to a csv
    try:
        df.to_csv(
            os.path.join(config["output_dir"], "skims_attached", skim_output_file),
            index=False,
        )
    except:
        print("failed on export of output")


def fetch_skim(
    df_name, df, time_field, mode_field, otaz_field, dtaz_field, use_mode=False
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
        df["hhincome"],
        bins=[-1, 84500, 108000, 9999999999],
        right=True,
        labels=[1, 2, 3],
        retbins=False,
        precision=3,
        include_lowest=True,
    )

    df["VOT Bin"] = df["VOT Bin"].astype("int")

    # Divide by 60 and round down to get hours (from minutes after midnight)
    hours = np.asarray(df[time_field].apply(lambda row: int(math.floor(row / 60))))
    df["dephr"] = [
        config["tod_dict"][hours.astype("str")[i]] for i in range(len(hours))
    ]

    # Look up mode keyword unless using standard mode value (e.g.)
    df[mode_field] = df[mode_field].fillna(-99)
    modes = np.asarray(df[mode_field].astype("int").astype("str"))
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
        final_df = final_df.append(tempdf)
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
    write_skims(df, skim_dict, otaz_field, dtaz_field, skim_output_file)
    # join skim data to original .dat files
    # Attach trip-level skim data to person records


def process_person_skims(tour, person, hh):
    """
    Add person and HH level data to trip records.
    """

    tour_per = pd.merge(tour, person, on=["hhno", "pno"], how="left")
    tour_per["unique_id"] = (
        tour_per.hhno.astype("str") + "_" + tour_per.pno.astype("str")
    )
    tour_per["unique_tour_id"] = (
        tour_per["unique_id"]
        + "_"
        + tour_per["day"].astype("str")
        + "_"
        + tour_per["tour"].astype("str")
    )

    # Use tour file to get work departure/arrival time and mode
    work_tours = tour_per[tour_per["pdpurp"] == 1]

    # Fill fields for usual work mode and times
    work_tours["puwmode"] = work_tours["tmodetp"]
    work_tours["puwarrp"] = work_tours["tardest"]
    work_tours["puwdepp"] = work_tours["tlvdest"]

    # some people make multiple work tours; select only the tours with greatest distance
    primary_work_tour = work_tours.groupby("unique_id")[
        "tlvdest", "unique_tour_id"
    ].max()
    work_tours = work_tours[
        work_tours.unique_tour_id.isin(primary_work_tour["unique_tour_id"].values)
    ]

    # drop the original Work Mode field
    person.drop(["puwmode", "puwarrp", "puwdepp"], axis=1, inplace=True)

    person = pd.merge(
        person,
        work_tours[["hhno", "pno", "puwmode", "puwarrp", "puwdepp"]],
        on=["hhno", "pno"],
        how="left",
    )

    # Fill NA for this field with -1
    for field in ["puwmode", "puwarrp", "puwdepp"]:
        person[field].fillna(-1, inplace=True)

    # Get school tour info
    # pusarrp and pusdepp are non-daysim variables, meaning usual arrival and departure time from school
    school_tours = tour_per[tour_per["pdpurp"] == 2]
    school_tours["pusarrp"] = school_tours["tardest"]
    school_tours["pusdepp"] = school_tours["tlvdest"]

    # Select a primary school trip, based on longest distance
    primary_school_tour = school_tours.groupby("unique_id")[
        "tlvdest", "unique_tour_id"
    ].max()
    school_tours = school_tours[
        school_tours["unique_tour_id"].isin(
            primary_school_tour["unique_tour_id"].values
        )
    ]

    person = pd.merge(
        person,
        school_tours[["hhno", "pno", "pusarrp", "pusdepp"]],
        on=["hhno", "pno"],
        how="left",
    )

    for field in ["pusarrp", "pusdepp"]:
        person[field].fillna(-1, inplace=True)

    # Attach hhincome and TAZ info
    person = pd.merge(person, hh[["hhno", "hhincome", "hhtaz"]], on="hhno", how="left")

    # Fill -1 income (college students) with lowest income category
    min_income = person[person["hhincome"] > 0]["hhincome"].min()
    person.loc[person["hhincome"] > 0, "hhincome"] = min_income

    # Convert fields to int
    for field in ["pusarrp", "pusdepp", "puwarrp", "puwdepp"]:
        person[field] = person[field].astype("int")

    # Write results to CSV for new derived fields
    person.to_csv(
        os.path.join(config["output_dir"], "skims_attached", "person_skim_output.csv"),
        index=False,
    )

    return person


def update_records(trip, tour, person):
    """
    Add skim value results to original survey files.
    """

    # Load skim data
    trip_skim = pd.read_csv(
        os.path.join(config["output_dir"], "skims_attached", "trip_skim_output.csv")
    )
    tour_skim = pd.read_csv(
        os.path.join(config["output_dir"], "skims_attached", "tour_skim_output.csv")
    )
    person_skim = pd.read_csv(
        os.path.join(config["output_dir"], "skims_attached", "person_skim_output.csv")
    )
    work_skim = pd.read_csv(
        os.path.join(
            config["output_dir"], "skims_attached", "work_travel_skim_output.csv"
        )
    )
    school_skim = pd.read_csv(
        os.path.join(
            config["output_dir"], "skims_attached", "school_travel_skim_output.csv"
        )
    )

    for df in [trip_skim, tour_skim, person_skim, work_skim, school_skim]:
        df["id"] = df["id"].astype("str")

    for df in [trip, tour, person]:
        df["id"] = df["id"].astype("str")

    trip_cols = {"travtime": "t", "travcost": "c", "travdist": "d"}
    tour_cols = {"tautotime": "t", "tautocost": "c", "tautodist": "d"}
    person_cols = {"puwmode": "puwmode", "puwarrp": "puwarrp", "puwdepp": "puwdepp"}
    work_cols = {"pwautime": "t", "pwaudist": "d"}
    school_cols = {"psautime": "t", "psaudist": "d"}

    # drop skim columns from the old file
    trip.drop(trip_cols.keys(), axis=1, inplace=True)
    tour.drop(tour_cols.keys(), axis=1, inplace=True)
    person.drop(person_cols.keys(), axis=1, inplace=True)
    person.drop(work_cols.keys(), axis=1, inplace=True)
    person.drop(school_cols.keys(), axis=1, inplace=True)

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
        os.path.join(config["output_dir"], "skims_attached", "_trip.tsv"),
        index=False,
        sep="\t",
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
        os.path.join(config["output_dir"], "skims_attached", "_tour.tsv"),
        index=False,
        sep="\t",
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
        os.path.join(config["output_dir"], "skims_attached", "_person.tsv"),
        index=False,
        sep="\t",
    )


def dat_to_h5(file_list):
    group_dict = {
        "household_day": "HouseholdDay",
        "household": "Household",
        "person_day": "PersonDay",
        "person": "Person",
        "tour": "Tour",
        "trip": "Trip",
    }

    # Create H5 container (overwrite if exists)
    if os.path.isfile(os.path.join(config["output_dir"], "survey.h5")):
        os.remove(os.path.join(config["output_dir"], "survey.h5"))
    f = h5py.File(os.path.join(config["output_dir"], "survey.h5"), "w")

    # Process all csv files in this directory
    for fname in file_list:
        df = pd.read_csv(fname, sep="\t")
        df = df.fillna(-1)

        # # Create new group name based on CSV file name
        group_name = [group_dict[i] for i in group_dict.keys() if i in fname][0]
        grp = f.create_group(group_name)

        for column in df.columns:
            if column in [
                "travdist",
                "travcost",
                "travtime",
                "trexpfac",
                "tautotime",
                "tautocost",
                "tautodist",
                "toexpfac",
                "hdexpfac" "pwautime",
                "pwaudist",
                "psautime",
                "psaudist",
                "psexpfac",
                "pdexpfac",
                "hhexpfac",
            ]:
                grp.create_dataset(column, data=list(df[column].astype("float64")))
            else:
                grp.create_dataset(column, data=list(df[column].astype("int32")))

        print("Added to h5 container: " + str(group_name))

    f.close()


def hhmm_to_mam(df, field):
    """
    Convert time in HHMM format to minutes after midnight.
    """

    # Strip minutes and seconds fields HHMM format
    hr = df[field].astype("str").apply(lambda row: row[: len(row) - 2]).astype("int")
    minute = (
        df[field].astype("str").apply(lambda row: row[len(row) - 2 :]).astype("int")
    )

    # Hours range from 3-27; if any are greater than 24, subtract 24 so the range goes from 0-24
    hr = pd.DataFrame(hr)
    hr.loc[hr[field] >= 24, field] = hr.loc[hr[field] >= 24, field] - 24

    df[field] = (hr[field] * 60) + minute

    return df


def attach_skims():
    # Create new output directory for survey records with skims attached, if needed
    if not os.path.exists(os.path.join(config["output_dir"], "skims_attached")):
        os.makedirs(os.path.join(config["output_dir"], "skims_attached"))

    # Load daysim-converted data produced from daysim_conversion.py
    trip = pd.read_csv(os.path.join(config["output_dir"], "_trip.tsv"), sep="\t")
    tour = pd.read_csv(os.path.join(config["output_dir"], "_tour.tsv"), sep="\t")
    hh = pd.read_csv(os.path.join(config["output_dir"], "_household.tsv"), sep="\t")
    person = pd.read_csv(os.path.join(config["output_dir"], "_person.tsv"), sep="\t")

    # Drop any rows with -1 expansion factor
    _filter = person["psexpfac"] >= 0
    logger.info(f"Dropped {len(person[~_filter])} persons: -1 expansion factor")
    person = person[_filter]

    # Add unique id fields
    person["id"] = person["hhno"].astype("str") + person["pno"].astype("str")
    trip["id"] = (
        trip["hhno"].astype("str")
        + trip["pno"].astype("str")
        + trip["tour"].astype("str")
        + trip["half"].astype("str")
        + trip["tseg"].astype("str")
    )
    tour["id"] = (
        tour["hhno"].astype("str")
        + tour["pno"].astype("str")
        + tour["tour"].astype("str")
    )

    # Join household to trip data to get income
    trip_hh = pd.merge(trip, hh, on="hhno")
    tour_hh = pd.merge(tour, hh, on="hhno")

    # Extract person-level results from trip file
    person_modified = process_person_skims(tour, person, hh)

    # Fetch trip skims based on trip departure time
    fetch_skim(
        "trip",
        trip_hh,
        time_field="deptm",
        mode_field="mode",
        otaz_field="otaz",
        dtaz_field="dtaz",
    )

    # Fetch tour skims based on tour departure time from origin
    fetch_skim(
        "tour",
        tour_hh,
        time_field="tlvorig",
        mode_field="tmodetp",
        otaz_field="totaz",
        dtaz_field="tdtaz",
    )

    # Attach person-level work skims based on home to work auto trips
    fetch_skim(
        "work_travel",
        person_modified,
        time_field="puwarrp",
        mode_field="puwmode",
        otaz_field="hhtaz",
        dtaz_field="pwtaz",
        use_mode="3",
    )

    # Attach person-level school skims based on home to school auto trips
    fetch_skim(
        "school_travel",
        person_modified,
        time_field="pusarrp",
        mode_field="puwmode",
        otaz_field="hhtaz",
        dtaz_field="pstaz",
        use_mode="3",
    )

    # Reload original person file and attach skim results
    person = pd.read_csv(os.path.join(config["output_dir"], "_person.tsv"), sep="\t")
    person["id"] = person["hhno"].astype("str") + person["pno"].astype("str")

    # Update records
    update_records(trip, tour, person)

    # Write results to h5
    write_list = ["household", "person_day", "household_day"]

    # Copy non-updated files to the same output directory for consistency
    for file in write_list:
        df = pd.read_csv(
            os.path.join(config["output_dir"], "_" + file + ".tsv"), sep="\t"
        )
        df.to_csv(
            os.path.join(config["output_dir"], "skims_attached", "_" + file + ".tsv"),
            sep="\t",
            index=False,
        )

    dat_to_h5(
        [
            os.path.join(config["output_dir"], "skims_attached", "_" + file + ".tsv")
            for file in write_list
        ]
    )
