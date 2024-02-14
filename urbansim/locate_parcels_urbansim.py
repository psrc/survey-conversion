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

config = toml.load("urbansim/urbansim_configuration.toml")

logger = logcontroller.setup_custom_logger("urbansim/locate_parcels_logger.txt")
logger.info("--------------------locate_parcels.py STARTING--------------------")
start_time = datetime.datetime.now()


def locate_parcels():
    if config["use_elmer"]:
        hh_original = util.load_elmer_table(config["elmer_hh_table"])
    else:
        hh_original = pd.read_csv(os.path.join(config["survey_input_dir"], "hh.csv"))

    # Load parcel data
    parcel_df = pd.read_csv(config["parcel_file_dir"], delim_whitespace=True)

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
        hh_original.copy(), parcel_df, hh_filter_dict_list
    )

    # Write to file
    hh_new.rename(columns={"hhid": "household_id"}, inplace=True)
    hh_new.to_csv(os.path.join(config["output_dir"], "geolocated_hh.csv"), index=False)

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------locate_parcels.py ENDING--------------------")
    logger.info("locate_parcels.py RUN TIME %s" % str(elapsed_total))
