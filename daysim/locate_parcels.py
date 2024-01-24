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
import configuration
from modules import geolocate, util

pd.options.mode.chained_assignment = None  # default='warn'

config = toml.load("daysim_configuration.toml")

logger = logcontroller.setup_custom_logger("locate_parcels_logger.txt")
logger.info("--------------------locate_parcels.py STARTING--------------------")
start_time = datetime.datetime.now()

def locate_parcels():
    if config["use_elmer"]:
        trip_original = util.load_elmer_table("HHSurvey.v_trips", config["survey_year"])
        person_original = util.load_elmer_table("HHSurvey.v_persons", config["survey_year"])
        hh_original = util.load_elmer_table("HHSurvey.v_households", config["survey_year"])
    else:
        trip_original = pd.read_csv(
            os.path.join(config["survey_input_dir"], "trip.csv")
        )
        person_original = pd.read_csv(
            os.path.join(config["survey_input_dir"], "person.csv")
        )
        hh_original = pd.read_csv(os.path.join(config["survey_input_dir"], "hh.csv"))

    # Load parcel data
    parcel_df = pd.read_csv(config["parcel_file_dir"], delim_whitespace=True)
    parcel_maz_df = pd.read_csv(config["parcel_maz_file_dir"])

    # Join MAZ data to parcel records
    parcel_df = parcel_df.merge(parcel_maz_df, left_on='parcelid', right_on='parcel_id')

    ##################################################
    # Process Household Records
    ##################################################

    hh_filter_dict_list = [
        {
            # Current Home Location
            "var_name": "final_home",
            "parcel_filter": (parcel_df["hh_p"] > 0),
            "hh_filter": (-hh_original["final_home_lat"].isnull()),
        },
        {
            # Previous Home Location
            "var_name": "prev_home",
            "parcel_filter": (parcel_df["hh_p"] > 0),
            "hh_filter": (-hh_original["prev_home_lat"].isnull()),
        },
    ]

    hh_new = geolocate.locate_hh_parcels(hh_original.copy(), parcel_df, hh_filter_dict_list)

    # Write to file
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
                "final_home_lat",
                "final_home_lng",
                "final_home_parcel",
                "final_home_taz",
                "final_home_maz",
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
            "parcel_filter": parcel_df["emptot_p"] > 0,
            "person_filter": (-person["work_lng"].isnull())
            & (
                person["workplace"] == "Usually the same location (outside home)"
            ),  # workplace is at a consistent location
        },
        {
            # Previous work location
            "var_name": "prev_work",
            "parcel_filter": parcel_df["emptot_p"] > 0,
            "person_filter": -person["prev_work_lng"].isnull(),
        },
        {
            # Student
            "var_name": "school_loc",
            "parcel_filter": (parcel_df["total_students"] > 0),
            "person_filter": (-person["school_loc_lat"].isnull())
            & (-(person["final_home_lat"] == person["school_loc_lat"]))
            & (  # Exclude home-school students
                -(person["final_home_lng"] == person["school_loc_lng"])
            ),
        },
    ]

    person = geolocate.locate_person_parcels(person, parcel_df, filter_dict_list)

    # For people that work from home, assign work parcel as household parcel
    # Join this person file back to original person file to get workplace
    for geog in ['parcel','taz','maz']:
        person.loc[
            person["workplace"].isin(config["usual_workplace_home"]), "work_"+geog
        ] = person["final_home_"+geog]

    person_loc_fields = [
        "school_loc_parcel",
        "school_loc_taz",
        "school_loc_maz",
        "work_parcel",
        "work_taz",
        "work_maz",
        "prev_work_parcel",
        "prev_work_taz",
        "prev_work_maz",
        "school_loc_parcel_distance",
        "work_parcel_distance",
        "prev_work_parcel_distance",
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

    ##################################################
    # Process Trip Records
    ##################################################

    opurp_field = "origin_purpose"
    dpurp_field = "dest_purpose"

    trip_results = trip_original.copy()

# Dictionary of filters to be applied
    # Filters are by trip purpose and define which parcels should be available for selection as nearest
    trip_filter_dict_list = [
        # Home trips (purp == 1) should be nearest parcel with household population > 0
        {
            "parcel_filter": parcel_df["hh_p"] > 0,
            "o_trip_filter": trip_results[opurp_field] == "Went home",
            "d_trip_filter": trip_results[dpurp_field] == "Went home",
        },
        # Work trips (purp.isin([10,11,14]), parcel must have jobs (emptot>0)
        {
            "parcel_filter": parcel_df["emptot_p"] > 0,
            "o_trip_filter": trip_results[opurp_field].isin(
                [
                    "Went to work-related place (e.g., meeting, second job, delivery)",
                    "Went to primary workplace",
                    "Went to other work-related activity",
                ]
            ),
            "d_trip_filter": trip_results[dpurp_field].isin(
                [
                    "Went to work-related place (e.g., meeting, second job, delivery)",
                    "Went to primary workplace",
                    "Went to other work-related activity",
                ]
            ),
        },
        # School (purp==6); parcel must have students (either grade, high, or uni students)
        {
            "parcel_filter": (
                (parcel_df["stugrd_p"] > 0)
                | (parcel_df["stuhgh_p"] > 0)
                | (parcel_df["stuuni_p"] > 0)
            ),
            "o_trip_filter": trip_results[opurp_field]
            == "Went to school/daycare (e.g., daycare, K-12, college)",
            "d_trip_filter": trip_results[dpurp_field]
            == "Went to school/daycare (e.g., daycare, K-12, college)",
        },
        # Escort (purp==9); parcel must have jobs or grade/high school students
        {
            "parcel_filter": (
                (parcel_df["stugrd_p"] > 0)
                | (parcel_df["stuhgh_p"] > 0)
                | (parcel_df["emptot_p"] > 0)
            ),
            "o_trip_filter": trip_results[opurp_field]
            == "Dropped off/picked up someone (e.g., son at a friend's house, spouse at bus stop)",
            "d_trip_filter": trip_results[dpurp_field]
            == "Dropped off/picked up someone (e.g., son at a friend's house, spouse at bus stop)",
        },
        # Personal Business/other apporintments, errands; parcel must have either retail or service jobs
        {
            "parcel_filter": (
                (parcel_df["empret_p"] > 0) | (parcel_df["empsvc_p"] > 0)
            ),
            "o_trip_filter": trip_results[opurp_field].isin(
                [
                    "Conducted personal business (e.g., bank, post office)",
                    "Other appointment/errands (rMove only)",
                ]
            ),
            "d_trip_filter": trip_results[dpurp_field].isin(
                [
                    "Conducted personal business (e.g., bank, post office)",
                    "Other appointment/errands (rMove only)",
                ]
            ),
        },
        # Shopping (purp.isin([30,32])); parcel must have retail jobs
        {
            "parcel_filter": parcel_df["empret_p"] > 0,
            "o_trip_filter": trip_results[opurp_field].isin(
                [
                    "Went grocery shopping",
                    "Went to other shopping (e.g., mall, pet store)",
                ]
            ),
            "d_trip_filter": trip_results[dpurp_field].isin(
                [
                    "Went grocery shopping",
                    "Went to other shopping (e.g., mall, pet store)",
                ]
            ),
        },
        # Meal (purp==50); parcel must have food service jobs; FIXME: maybe allow retail/other for things like grocery stores
        {
            "parcel_filter": parcel_df["empfoo_p"] > 0,
            "o_trip_filter": trip_results[opurp_field]
            == "Went to restaurant to eat/get take-out",
            "d_trip_filter": trip_results[dpurp_field]
            == "Went to restaurant to eat/get take-out",
        },
        # Social; parcel must have households or employment
        {
            "parcel_filter": (parcel_df["hh_p"] > 0),
            "o_trip_filter": trip_results[opurp_field].isin(
                [
                    "Attended social event (e.g., visit with friends, family, co-workers)",
                    "Other social/leisure (rMove only)",
                ]
            ),
            "d_trip_filter": trip_results[dpurp_field].isin(
                [
                    "Attended social event (e.g., visit with friends, family, co-workers)",
                    "Other social/leisure (rMove only)",
                ]
            ),
        },
        # Recreational, exercise, volunteed/community event, family activity, other (purp.isin([53,51,54]); any parcel allowed
        {
            "parcel_filter": -parcel_df.isnull(),  # no parcel filter
            "o_trip_filter": trip_results[opurp_field].isin(
                [
                    "Went to religious/community/volunteer activity",
                    "Attended recreational event (e.g., movies, sporting event)",
                    "Went to a family activity (e.g., child's softball game)",
                    "Went to exercise (e.g., gym, walk, jog, bike ride)",
                ]
            ),
            "d_trip_filter": trip_results[dpurp_field].isin(
                [
                    "Went to religious/community/volunteer activity",
                    "Attended recreational event (e.g., movies, sporting event)",
                    "Went to a family activity (e.g., child's softball game)",
                    "Went to exercise (e.g., gym, walk, jog, bike ride)",
                ]
            ),
        },
        # Medical; parcel must have medical employment
        {
            "parcel_filter": parcel_df["empmed_p"] > 0,
            "o_trip_filter": trip_results[opurp_field]
            == "Went to medical appointment (e.g., doctor, dentist)",
            "d_trip_filter": trip_results[dpurp_field]
            == "Went to medical appointment (e.g., doctor, dentist)",
        },
        # For change mode, no parcel filter
        {
            "parcel_filter": -parcel_df.isnull(),  # no parcel filter
            "o_trip_filter": trip_results[opurp_field]
            == "Transferred to another mode of transportation (e.g., change from ferry to bus)",
            "d_trip_filter": trip_results[dpurp_field]
            == "Transferred to another mode of transportation (e.g., change from ferry to bus)",
        },
    ]

    trip = geolocate.locate_trip_parcels(trip_original.copy(), parcel_df, opurp_field, dpurp_field, trip_filter_dict_list)

    # Export original survey records
    # Merge with originals to make sure we didn't exclude records
    trip_original_updated = trip_original.merge(
        trip[
            [
                "trip_id",
                "otaz",
                "dtaz",
                "opcl",
                "dpcl",
                "opcl_distance",
                "dpcl_distance",
            ]
        ],
        on="trip_id",
        how="left",
    )
    trip_original_updated["otaz"].fillna(-1, inplace=True)

    # Join other geography (MAZ)

    # Write to file
    trip_original_updated.to_csv(
        os.path.join(config["output_dir"], "geolocated_trip.csv"), index=False
    )

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------locate_parcels.py ENDING--------------------")
    logger.info("locate_parcels.py RUN TIME %s" % str(elapsed_total))