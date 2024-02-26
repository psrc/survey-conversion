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

# This script assigns parcel IDs to survey fields, including current/past work and home location,
# school location, and trip origin and destination.
# Locations are assigned by finding nearest parcel that meets criteria (e.g., work location must have workers at parcel)
# In some cases (school location), if parcels are over a threshold distance from original xy values,
# multiple filter tiers can be applied (e.g., first find parcel with students; for parcels with high distances,
# use a looser criteria like a parcel with service jobs, followed by parcels with household population.)

import os, sys
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pysal.lib.weights.distance import get_points_array
from shapely.geometry import LineString
from pyproj import Proj, transform
import pyproj
import numpy as np
from operator import itemgetter
import urllib
import pyodbc
import sqlalchemy
from sqlalchemy.engine import URL
from pymssql import connect
from shapely import wkt
import logging
from daysim import logcontroller
import datetime
import toml
from modules import geolocate, util

pd.options.mode.chained_assignment = None  # default='warn'


def locate_parcels_urbansim(config):
    logger = logcontroller.setup_custom_logger("locate_parcels_logger.txt", config)
    logger.info("--------------------locate_parcels.py STARTING--------------------")
    start_time = datetime.datetime.now()

    if config["use_elmer"]:
        hh_original = util.load_elmer_table(config["elmer_hh_table"])
        person_original = util.load_elmer_table(config["elmer_person_table"])
    else:
        hh_original = pd.read_csv(os.path.join(config["survey_input_dir"], "hh.csv"))
        person_original = pd.read_csv(
            os.path.join(config["survey_input_dir"], "person.csv")
        )

    # Load parcel data with residential unit data
    parcel_df = pd.read_csv(config["parcel_file_dir"])
    parcel_df.rename(columns={"parcel_id": "parcelid"}, inplace=True)
    parcel_df.drop(["xcoord_p", "ycoord_p"], inplace=True, axis=1)

    # Join with usual parcel data file that includes jobs and students
    parcel_df_standard = pd.read_csv(
        config["parcel_file_standard_dir"], delim_whitespace=True
    )
    parcel_df = parcel_df.merge(parcel_df_standard, on="parcelid", how="left")

    # Script expects MAZ and TAZ data
    parcel_maz_df = pd.read_csv(config["parcel_maz_file_dir"])
    parcel_df = parcel_df.merge(parcel_maz_df, left_on="parcelid", right_on="parcel_id")

    ##################################################
    # Process Household Records
    ##################################################

    hh_filter_dict_list = [
        {
            # Place current home location based on presence of residential units
            "var_name": "home",
            "parcel_filter": (parcel_df["residential_units"] > 0),
            "hh_filter": (-hh_original["home_lat"].isnull()),
        }
    ]

    hh_new = geolocate.locate_hh_parcels(
        hh_original.copy(), parcel_df, hh_filter_dict_list, config
    )

    # Write to file
    hh_new.rename(columns={"hhid": "household_id"}, inplace=True)
    hh_new.to_csv(os.path.join(config["output_dir"], "geolocated_hh.csv"), index=False)

    ###################################################
    # Process Person Records
    ###################################################

    # Merge with household records to get school/work lat and long, to filter people who home school and work at home
    person = pd.merge(
        person_original,
        hh_new[
            [
                "household_id",
                "home_lat",
                "home_lng",
                "home_parcel",
                "home_taz",
                "home_maz",
            ]
        ],
        on="household_id",
    )

    # Add parcel location for current and previous school and workplace location
    parcel_df["total_students"] = parcel_df[["stugrd_p", "stuhgh_p", "stuuni_p"]].sum(
        axis=1
    )

    # Find parcels for person fields
    filter_dict_list = [
        {
            "var_name": "work",
            "parcel_filter": (parcel_df["emptot_p"] > 0)
            & (parcel_df["number_of_buildings"] > 0),
            "person_filter": (-person["work_lng"].isnull())
            & (  # workplace is at a consistent location
                person["workplace"] == "Usually the same location (outside home)"
            ),
        },
        {
            # Student
            "var_name": "school_loc",
            "parcel_filter": (parcel_df["total_students"] > 0)
            & (parcel_df["number_of_buildings"] > 0),
            "person_filter": (-person["school_loc_lat"].isnull())
            & (-(person["home_lat"] == person["school_loc_lat"]))
            & (  # Exclude home-school students
                -(person["home_lng"] == person["school_loc_lng"])
            ),
        },
    ]

    person = geolocate.locate_person_parcels(
        person, parcel_df, filter_dict_list, config
    )

    # For people that work from home, assign work parcel as household parcel
    # Join this person file back to original person file to get workplace
    for geog in ["parcel", "taz", "maz"]:
        person.loc[
            person["workplace"].isin(config["usual_workplace_home"]), "work_" + geog
        ] = person["home_" + geog]

    person_loc_fields = [
        "school_loc_parcel",
        "school_loc_taz",
        "school_loc_maz",
        "work_parcel",
        "work_taz",
        "work_maz",
        "school_loc_parcel_distance",
        "work_parcel_distance",
    ]

    # Join selected fields back to the original person file
    person_orig_update = person_original.merge(
        person[person_loc_fields + ["person_id"]], on="person_id", how="left"
    )
    person_orig_update[person_loc_fields] = (
        person_orig_update[person_loc_fields].fillna(-1).astype("int")
    )

    # Write to file
    person_orig_update.to_csv(
        os.path.join(config["output_dir"], "geolocated_person.csv"), index=False
    )

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------locate_parcels.py ENDING--------------------")
    logger.info("locate_parcels.py RUN TIME %s" % str(elapsed_total))
