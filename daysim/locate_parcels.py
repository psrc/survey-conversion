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


import os, sys
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
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


def locate_parcels(config):
    logger = logcontroller.setup_custom_logger("locate_parcels_logger.txt", config)
    logger.info("--------------------locate_parcels.py STARTING--------------------")
    start_time = datetime.datetime.now()

    if config["use_elmer"]:
        trip_original = util.load_elmer_table(config["elmer_trip_table"], 
                                              sql="SELECT * FROM "+config["elmer_trip_table"]+\
                                                  " WHERE survey_year in "+str(config['survey_year']))
        hh_original = util.load_elmer_table(config["elmer_hh_table"], 
                                              sql="SELECT * FROM "+config["elmer_hh_table"]+\
                                                  " WHERE survey_year in "+str(config['survey_year']))
        person_original = util.load_elmer_table(config["elmer_person_table"], 
                                              sql="SELECT * FROM "+config["elmer_person_table"]+\
                                                  " WHERE survey_year in "+str(config['survey_year']))
    else:
        trip_original = pd.read_csv(
            os.path.join(config["survey_input_dir"], "trip.csv")
        )
        person_original = pd.read_csv(
            os.path.join(config["survey_input_dir"], "person.csv")
        )
        hh_original = pd.read_csv(os.path.join(config["survey_input_dir"], "hh.csv"))

    # if "trip_id" not in trip_original.columns:
    #     trip_original['trip_id'] = range(1,len(trip_original)+1)
    trip_original.set_index("trip_id", inplace=True)

    # Shorten the home lat/lng field names
    # FIXME: change this in Elmer directly
    hh_original.rename(columns={'final_home_lat': 'home_lat',
                                  'final_home_lng': 'home_lng'}, inplace=True)


    # Load parcel data
    parcel_df = pd.read_csv(config["parcel_file_dir"], delim_whitespace=True)
    parcel_maz_df = pd.read_csv(config["parcel_maz_file_dir"])

    # Join MAZ data to parcel records
    parcel_df = parcel_df.merge(parcel_maz_df, left_on="parcelid", right_on="parcel_id")

    ##################################################
    # Process Household Records
    ##################################################

    hh_filter_dict_list = [
        {
            # Current Home Location
            "var_name": "home",
            "parcel_filter": (parcel_df["hh_p"] > 0),
            "hh_filter": (-hh_original["home_lat"].isnull()),
        }
    ]

    hh_new = geolocate.locate_hh_parcels(
        hh_original.copy(), parcel_df, hh_filter_dict_list, config
    )

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
            "parcel_filter": parcel_df["emptot_p"] > 0,
            "person_filter": (-person["work_lng"].isnull()),
        },
        {
            # Student
            "var_name": "school_loc",
            "parcel_filter": (parcel_df["total_students"] > 0),
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

    # # For people that work from home, assign work parcel as household parcel
    # # Join this person file back to original person file to get workplace
    # for geog in ["parcel", "taz", "maz"]:
    #     person.loc[
    #         person["workplace"].isin(config["usual_workplace_home"]), "work_" + geog
    #     ] = person["home_" + geog]

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

    ##################################################
    # Process Trip Records
    ##################################################

    trip_results = trip_original.copy()

    df_lookup = pd.read_csv(os.path.join(config["input_dir"], "data_lookup.csv"))
    df_lookup = df_lookup[df_lookup["table"] == "trip"]
    df_lookup_o = df_lookup[df_lookup["elmer_name"] == config['opurp_field']]
    df_lookup_d = df_lookup[df_lookup["elmer_name"] == config['dpurp_field']]

    # filter out some odd results with lng > 0 and lat < 0
    for trip_end in ["origin", "dest"]:
        lng_field = trip_end + "_lng"
        lat_field = trip_end + "_lat"

        # filter out some odd results with lng > 0 and lat < 0
        _filter = trip_results[lat_field] > 0
        logger.info(
            f"Dropped {len(trip_results[~_filter])} trips: " + trip_end + " lat > 0 "
        )
        trip_results = trip_results[_filter]

        _filter = trip_results[lng_field] < 0
        logger.info(
            f"Dropped {len(trip_results[~_filter])} trips: " + trip_end + " lng < 0 "
        )
        trip_results = trip_results[_filter]

    # trip_results = trip_results.reset_index()

    # Dictionary of filters to be applied
    # Filters are by trip purpose and define which parcels should be available for selection as nearest
    trip_filter_dict_list = [
        # Home trips (purp == 0) should be nearest parcel with household population > 0
        {
            "type": "home",
            "parcel_filter": parcel_df["hh_p"] > 0,
            "o_trip_filter": trip_results[config["opurp_field"]].isin(
                df_lookup_o.loc[df_lookup_o["model_value"] == 0, "elmer_value"]
            ),
            "d_trip_filter": trip_results[config["dpurp_field"]].isin(
                df_lookup_d.loc[df_lookup_d["model_value"] == 0, "elmer_value"]
            ),
        },
        # Work and work-related trips parcels must have jobs (emptot>0)
        {
            "type": "work",
            "parcel_filter": parcel_df["emptot_p"] > 0,
            "o_trip_filter": trip_results[config["opurp_field"]].isin(
                df_lookup_o.loc[df_lookup_o["model_value"] == 1, "elmer_value"]
            ),
            "d_trip_filter": trip_results[config["dpurp_field"]].isin(
                df_lookup_d.loc[df_lookup_d["model_value"] == 1, "elmer_value"]
            ),
        },
        # School (purp==2); parcel must have students (either grade, high, or uni students)
        {
            "type": "school",
            "parcel_filter": (
                (parcel_df["stugrd_p"] > 0)
                | (parcel_df["stuhgh_p"] > 0)
                | (parcel_df["stuuni_p"] > 0)
            ),
            "o_trip_filter": trip_results[config["opurp_field"]].isin(
                df_lookup_o.loc[df_lookup_o["model_value"] == 2, "elmer_value"]
            ),
            "d_trip_filter": trip_results[config["dpurp_field"]].isin(
                df_lookup_d.loc[df_lookup_d["model_value"] == 2, "elmer_value"]
            ),
        },
        # Personal Business/other apporintments, errands; parcel must have either retail or service jobs
        {
            "type": "personal business",
            "parcel_filter": (
                (parcel_df["empret_p"] > 0) | (parcel_df["empsvc_p"] > 0)
            ),
            "o_trip_filter": trip_results[config["opurp_field"]].isin(
                df_lookup_o.loc[df_lookup_o["model_value"] == 4, "elmer_value"]
            ),
            "d_trip_filter": trip_results[config["dpurp_field"]].isin(
                df_lookup_d.loc[df_lookup_d["model_value"] == 4, "elmer_value"]
            ),
        },
        # Shopping (purp.isin([30,32])); parcel must have retail jobs
        {
            "type": "shopping",
            "parcel_filter": parcel_df["empret_p"] > 0,
            "o_trip_filter": trip_results[config["opurp_field"]].isin(
                df_lookup_o.loc[df_lookup_o["model_value"] == 5, "elmer_value"]
            ),
            "d_trip_filter": trip_results[config["dpurp_field"]].isin(
                df_lookup_d.loc[df_lookup_d["model_value"] == 5, "elmer_value"]
            ),
        },
        # Meal (purp==50); parcel must have food service jobs; FIXME: maybe allow retail/other for things like grocery stores
        {
            "type": "meal",
            "parcel_filter": parcel_df["empret_p"] > 0,
            "o_trip_filter": trip_results[config["opurp_field"]].isin(
                df_lookup_o.loc[df_lookup_o["model_value"] == 6, "elmer_value"]
            ),
            "d_trip_filter": trip_results[config["dpurp_field"]].isin(
                df_lookup_d.loc[df_lookup_d["model_value"] == 6, "elmer_value"]
            ),
        },
        # Social, recreational, change mode, escort, missing, no constraints
        {
            "type": "other",
            "parcel_filter": -parcel_df.isnull(),
            "o_trip_filter": trip_results[config["opurp_field"]].isin(
                df_lookup_o.loc[
                    df_lookup_o["model_value"].isin([3, 7, 10]), "elmer_value"
                ]
            ),
            "d_trip_filter": trip_results[config["dpurp_field"]].isin(
                df_lookup_d.loc[
                    df_lookup_d["model_value"].isin([3, 7, 10]), "elmer_value"
                ]
            ),
        },
    ]

    trip = geolocate.locate_trip_parcels(
        trip_results,
        parcel_df,
        config["opurp_field"],
        config["dpurp_field"],
        trip_filter_dict_list,
        config,
    )

    # Export original survey records
    # Merge with originals to make sure we didn't exclude records
    trip_original_updated = trip_original.merge(
        trip[
            [
                "otaz",
                "dtaz",
                "opcl",
                "dpcl",
                "opcl_distance",
                "dpcl_distance",
            ]
        ],
        left_index=True,
        right_index=True,
        how="left",
    )
    trip_original_updated["otaz"].fillna(-1, inplace=True)
    trip_original_updated["dtaz"].fillna(-1, inplace=True)

    trip_original_updated["trip_id"] = trip_original_updated.index

    # Write to file
    trip_original_updated.to_csv(
        os.path.join(config["output_dir"], "geolocated_trip.csv"), index=False
    )

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------locate_parcels.py ENDING--------------------")
    logger.info("locate_parcels.py RUN TIME %s" % str(elapsed_total))
