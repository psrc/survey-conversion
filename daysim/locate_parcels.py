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
import logcontroller
import datetime
import toml
import configuration

pd.options.mode.chained_assignment = None  # default='warn'

config = toml.load("configuration.toml")

logger = logcontroller.setup_custom_logger("locate_parcels_logger.txt")
logger.info("--------------------locate_parcels.py STARTING--------------------")
start_time = datetime.datetime.now()


def nearest_neighbor(df_parcel_coord, df_trip_coord):
    """Find 1st nearest parcel location for trip location
    df_parcel: x and y columns of parcels
    df_trip: x and y columns of trip records

    Returns: tuple of distance between nearest points and index of df_parcel for nearest parcel

    """

    kdt_parcels = cKDTree(df_parcel_coord)
    return kdt_parcels.query(df_trip_coord, k=1)


def locate_parcel(
    _parcel_df, df, xcoord_col, ycoord_col, parcel_filter=None, df_filter=None
):
    """Find nearest parcel to XY coordinate. Returns distance between parcel node and points, and selected parcel ID

    Inputs:
    - _parcel_df: full parcels dataset
    - df: records to be located
    - xcoord_col and ycood_col: column names of x and y coordinates
    - parcel_filter: filter to be applied to _parcels_df to generate candidate parcels
    - df_filter: filter on point-level data

    Returns: 2 lists: (1) distance between nearest parcel [_dist] and (2) index of nearest parcel [_ix]
    """

    if parcel_filter is not None:
        _parcel_df = _parcel_df[parcel_filter].reset_index()
    if df_filter is not None:
        df = df[df_filter].reset_index()

    # Calculate distance to nearest neighbor parcel
    (
        _dist,
        _ix,
    ) = nearest_neighbor(
        _parcel_df[["xcoord_p", "ycoord_p"]], df[[xcoord_col, ycoord_col]]
    )

    return _dist, _ix


def locate_person_parcels(person, parcel_df, df_taz):
    """Locate parcel ID for school, workplace, home location from person records."""

    person_results = person.copy()  # Make local copy for storing resulting joins

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

    # Find nearest school and workplace
    for i in range(len(filter_dict_list)):
        varname = filter_dict_list[i]["var_name"]
        person_filter = filter_dict_list[i]["person_filter"]
        parcel_filter = filter_dict_list[i]["parcel_filter"]

        # Convert GPS Coordinates to State Plane
        gdf = gpd.GeoDataFrame(
            person[person_filter],
            geometry=gpd.points_from_xy(
                person[person_filter][varname + "_lng"],
                person[person_filter][varname + "_lat"],
            ),
        )
        gdf.crs = config["lat_lng_crs"]
        gdf[varname + "_lng_gps"] = gdf[varname + "_lng"]
        gdf[varname + "_lat_gps"] = gdf[varname + "_lat"]
        gdf = gdf.to_crs(config["wa_state_plane_crs"])  # convert to state plane WA

        # Spatial join between region taz file and person file
        gdf = gpd.sjoin(gdf, df_taz)
        xy_field = get_points_array(gdf.geometry)
        gdf[varname + "_lng_gps"] = gdf[varname + "_lng"]
        gdf[varname + "_lat_gps"] = gdf[varname + "_lat"]
        gdf[varname + "_lng_fips_4601"] = xy_field[:, 0]
        gdf[varname + "_lat_fips_4601"] = xy_field[:, 1]

        # Return: (_dist) the distance to the closest parcel that meets given critera,
        #         (_ix) list of the indices of parcel IDs from (_df), which is the filtered set of candidate parcels
        _dist, _ix = locate_parcel(
            parcel_df[parcel_filter],
            df=gdf,
            xcoord_col=varname + "_lng_fips_4601",
            ycoord_col=varname + "_lat_fips_4601",
        )

        # Assign values to person df, extracting from the filtered set of parcels (_df)
        gdf[varname + "_parcel"] = parcel_df[parcel_filter].iloc[_ix].parcelid.values
        gdf[varname + "_parcel_distance"] = _dist
        gdf[varname + "_taz"] = gdf["taz"]

        gdf_cols = [
            "person_id",
            varname + "_taz",
            varname + "_parcel",
            varname + "_parcel_distance",
            varname + "_lat_fips_4601",
            varname + "_lng_fips_4601",
            varname + "_lat_gps",
        ]

        # Refine School Location in 2 tiers
        # Tier 2: for locations that are over 1 mile (5280 feet) from lat/lng,
        # place them in parcel with >0 education or service employees (could be daycare or specialized school, etc. without students listed)

        if varname == "school_loc":
            hh_max_dist = 5280
            gdf_far = gdf[gdf[varname + "_parcel_distance"] > hh_max_dist]
            _dist, _ix = locate_parcel(
                parcel_df[parcel_df["total_students"] > 0],
                df=gdf_far,
                xcoord_col=varname + "_lng_fips_4601",
                ycoord_col=varname + "_lat_fips_4601",
            )
            gdf_far[varname + "_parcel"] = parcel_df.iloc[_ix].parcelid.values
            gdf_far[varname + "_parcel_distance"] = _dist
            gdf_far[varname + "_taz"] = gdf_far["taz"].astype("int")

            # Add this new distance to the original gdf
            gdf.loc[gdf_far.index, varname + "_parcel_original"] = gdf.loc[
                gdf_far.index, varname + "_parcel"
            ]
            gdf.loc[gdf_far.index, varname + "_parcel_distance_original"] = gdf.loc[
                gdf_far.index, varname + "_parcel_distance"
            ]
            gdf.loc[gdf_far.index, varname + "_parcel"] = gdf_far[varname + "_parcel"]
            gdf.loc[gdf_far.index, varname + "_parcel_distance"] = gdf_far[
                varname + "_parcel_distance"
            ]
            gdf["distance_flag"] = 0
            gdf.loc[gdf_far.index, varname + "distance_flag"] = 1

            gdf_cols += [
                varname + "_parcel_distance_original",
                varname + "_parcel_original",
            ]

        # Join the gdf dataframe to the person df
        person_results = person_results.merge(gdf[gdf_cols], how="left", on="person_id")

    # return person_results, person_daysim
    return person_results


def locate_hh_parcels(hh, parcel_df, df_taz):
    hh_results = hh.copy()

    filter_dict_list = [
        {
            # Current Home Location
            "var_name": "final_home",
            "parcel_filter": (parcel_df["hh_p"] > 0),
            "hh_filter": (-hh["final_home_lat"].isnull()),
        },
        {
            # Previous Home Location
            "var_name": "prev_home",
            "parcel_filter": (parcel_df["hh_p"] > 0),
            "hh_filter": (-hh["prev_home_lat"].isnull()),
        },
    ]

    # Find nearest school and workplace
    for i in range(len(filter_dict_list)):
        varname = filter_dict_list[i]["var_name"]
        parcel_filter = filter_dict_list[i]["parcel_filter"]
        hh_filter = filter_dict_list[i]["hh_filter"]

        # Convert GPS Coordinates to State Plane
        gdf = gpd.GeoDataFrame(
            hh[hh_filter],
            geometry=gpd.points_from_xy(
                hh[hh_filter][varname + "_lng"], hh[hh_filter][varname + "_lat"]
            ),
        )
        gdf.crs = config["lat_lng_crs"]
        gdf[varname + "_lng_gps"] = gdf[varname + "_lng"]
        gdf[varname + "_lat_gps"] = gdf[varname + "_lat"]
        gdf = gdf.to_crs(config["wa_state_plane_crs"])  # convert to state plane WA

        # Spatial join between region taz file and person file
        gdf = gpd.sjoin(gdf, df_taz)
        xy_field = get_points_array(gdf.geometry)
        gdf[varname + "_lng_gps"] = gdf[varname + "_lng"]
        gdf[varname + "_lat_gps"] = gdf[varname + "_lat"]
        gdf[varname + "_lng_fips_4601"] = xy_field[:, 0]
        gdf[varname + "_lat_fips_4601"] = xy_field[:, 1]

        # Return: (_dist) the distance to the closest parcel that meets given critera,
        # (_ix) list of the indices of parcel IDs from (_df), which is the filtered set of candidate parcels
        _dist, _ix = locate_parcel(
            parcel_df[parcel_filter],
            df=gdf,
            xcoord_col=varname + "_lng_fips_4601",
            ycoord_col=varname + "_lat_fips_4601",
        )

        # Assign values to person df, extracting from the filtered set of parcels (_df)
        gdf[varname + "_parcel"] = parcel_df[parcel_filter].iloc[_ix].parcelid.values
        gdf[varname + "_parcel_distance"] = _dist
        gdf[varname + "_taz"] = gdf["taz"].astype("int")

        # For households that are not reasonably near a parcel with households,
        # add them to the nearset unfiltered parcel and flag
        # Typically occurs with households living on military bases
        hh_max_dist = 2000
        gdf_far = gdf[gdf[varname + "_parcel_distance"] > hh_max_dist]
        _dist, _ix = locate_parcel(
            parcel_df,
            df=gdf_far,
            xcoord_col=varname + "_lng_fips_4601",
            ycoord_col=varname + "_lat_fips_4601",
        )
        gdf_far[varname + "_parcel"] = parcel_df.iloc[_ix].parcelid.values
        gdf_far[varname + "_parcel_distance"] = _dist
        gdf_far[varname + "_taz"] = gdf_far["taz"].astype("int")

        # Add this new distance to the original gdf
        gdf.loc[gdf_far.index, varname + "_parcel_original"] = gdf.loc[
            gdf_far.index, varname + "_parcel"
        ]
        gdf.loc[gdf_far.index, varname + "_parcel_distance_original"] = gdf.loc[
            gdf_far.index, varname + "_parcel_distance"
        ]
        gdf.loc[gdf_far.index, varname + "_parcel"] = gdf_far[varname + "_parcel"]
        gdf.loc[gdf_far.index, varname + "_parcel_distance"] = gdf_far[
            varname + "_parcel_distance"
        ]
        gdf["distance_flag"] = 0
        gdf.loc[gdf_far.index, varname + "distance_flag"] = 1

        # Join the gdf dataframe to the person df
        hh_results = hh_results.merge(
            gdf[
                [
                    "hhid",
                    varname + "_taz",
                    varname + "_parcel",
                    varname + "_parcel_distance",
                    varname + "_parcel_distance_original",
                    varname + "_lat_fips_4601",
                    varname + "_parcel_original",
                    varname + "_lng_fips_4601",
                    varname + "_lat_gps",
                ]
            ],
            on="hhid",
            how="left",
        )

    return hh_results


def locate_trip_parcels(trip, parcel_df, df_taz):
    """Attach parcel ID to trip origins and destinations."""

    opurp_field = "origin_purpose"
    dpurp_field = "dest_purpose"

    trip_results = trip.copy()

    for trip_end in ["origin", "dest"]:
        lng_field = trip_end + "_lng"
        lat_field = trip_end + "_lat"

        # filter out some odd results with lng > 0 and lat < 0
        _filter = trip[lat_field] > 0
        logger.info(f"Dropped {len(trip[~_filter])} trips: " + trip_end + " lat > 0 ")
        trip = trip[_filter]

        _filter = trip[lng_field] < 0
        logger.info(f"Dropped {len(trip[~_filter])} trips: " + trip_end + " lng < 0 ")
        trip = trip[_filter]

        gdf = gpd.GeoDataFrame(
            trip, geometry=gpd.points_from_xy(trip[lng_field], trip[lat_field])
        )
        gdf.crs = config["lat_lng_crs"]
        gdf = gdf.to_crs(config["wa_state_plane_crs"])  # convert to state plane WA

        # Spatial join between region taz file and trip file
        gdf = gpd.sjoin(gdf, df_taz)
        xy_field = get_points_array(gdf.geometry)
        gdf[trip_end + "_lng_gps"] = gdf[trip_end + "_lng"]
        gdf[trip_end + "_lat_gps"] = gdf[trip_end + "_lat"]
        gdf[trip_end + "_lng_fips_4601"] = xy_field[:, 0]
        gdf[trip_end + "_lat_fips_4601"] = xy_field[:, 1]
        gdf[trip_end + "_taz"] = gdf["taz"]
        trip_results = trip_results.merge(
            gdf[
                [
                    "trip_id",
                    trip_end + "_lng_gps",
                    trip_end + "_lat_gps",
                    trip_end + "_lng_fips_4601",
                    trip_end + "_lat_fips_4601",
                    trip_end + "_taz",
                ]
            ],
            on="trip_id",
        )

    # Dictionary of filters to be applied
    # Filters are by trip purpose and define which parcels should be available for selection as nearest
    filter_dict_list = [
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

    final_df = trip_results.copy()
    # Loop through each trip end type (origin or destination) and each trip purpose
    for trip_end_type in ["origin", "dest"]:
        df_temp = pd.DataFrame()
        for i in range(len(filter_dict_list)):
            trip_filter = filter_dict_list[i][trip_end_type[0] + "_trip_filter"]
            parcel_filter = filter_dict_list[i]["parcel_filter"]

            _df = trip_results[trip_filter]

            _dist, _ix = locate_parcel(
                parcel_df[parcel_filter],
                df=_df,
                xcoord_col=trip_end_type + "_lng_fips_4601",
                ycoord_col=trip_end_type + "_lat_fips_4601",
            )

            _df[trip_end_type[0] + "pcl"] = (
                parcel_df[parcel_filter].iloc[_ix].parcelid.values
            )
            _df[trip_end_type[0] + "pcl_distance"] = _dist
            _df[trip_end_type[0] + "taz"] = trip_results[trip_end_type + "_taz"]

            df_temp = df_temp.append(_df)
        # Join df_temp to final field for each trip type
        final_df = final_df.merge(
            df_temp[
                [
                    "trip_id",
                    trip_end_type[0] + "pcl",
                    trip_end_type[0] + "pcl_distance",
                    trip_end_type[0] + "taz",
                ]
            ],
            on="trip_id",
        )

    return final_df


def load_elmer_geo(con, table_name):
    """Load ElmerGeo feature class as geodataframe."""

    cursor = con.cursor()
    feature_class_name = table_name
    geo_col_stmt = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME="
        + "'"
        + feature_class_name
        + "'"
        + " AND DATA_TYPE='geometry'"
    )
    geo_col = str(pd.read_sql(geo_col_stmt, con).iloc[0, 0])
    query_string = (
        "SELECT *,"
        + geo_col
        + ".STGeometryN(1).ToString()"
        + " FROM "
        + feature_class_name
    )
    df = pd.read_sql(query_string, con)

    df.rename(columns={"": "geometry"}, inplace=True)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    return gdf


def locate_parcels():
    if config["use_elmer"]:
        conn_string = "DRIVER={ODBC Driver 17 for SQL Server}; SERVER=AWS-PROD-SQL\Sockeye; DATABASE=Elmer; trusted_connection=yes"
        sql_conn = pyodbc.connect(conn_string)
        params = urllib.parse.quote_plus(conn_string)
        engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

        trip_original = pd.read_sql(
            sql="SELECT * FROM HHSurvey.v_trips WHERE survey_year IN "
            + config["survey_year"],
            con=engine,
        )
        person_original = pd.read_sql(
            sql="SELECT * FROM HHSurvey.v_persons WHERE survey_year IN "
            + config["survey_year"],
            con=engine,
        )
        hh_original = pd.read_sql(
            sql="SELECT * FROM HHSurvey.v_households WHERE survey_year IN "
            + config["survey_year"],
            con=engine,
        )
    else:
        trip_original = pd.read_csv(
            os.path.join(config["survey_input_dir"], "trip.csv")
        )
        person_original = pd.read_csv(
            os.path.join(config["survey_input_dir"], "person.csv")
        )
        hh_original = pd.read_csv(os.path.join(config["survey_input_dir"], "hh.csv"))

    # Load TAZ shapefile from ElmerGeo
    con = connect("AWS-Prod-SQL\Sockeye", database="ElmerGeo")
    df_taz = load_elmer_geo(con, "taz2010")
    con.close()
    df_taz.crs = config["wa_state_plane_crs"]

    # Load parcel data
    parcel_df = pd.read_csv(config["parcel_file_dir"], delim_whitespace=True)

    ##################################################
    # Process Household Records
    ##################################################

    hh_new = locate_hh_parcels(hh_original.copy(), parcel_df, df_taz)

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
            ]
        ],
        on="household_id",
    )

    # Add parcel location for current and previous school and workplace location
    person = locate_person_parcels(person, parcel_df, df_taz)

    # For people that work from home, assign work parcel as household parcel
    # Join this person file back to original person file to get workplace
    person.loc[
        person["workplace"].isin(config["usual_workplace_home"]), "work_parcel"
    ] = person["final_home_parcel"]
    person.loc[
        person["workplace"].isin(config["usual_workplace_home"]), "work_taz"
    ] = person["final_home_taz"]

    person_loc_fields = [
        "school_loc_parcel",
        "school_loc_taz",
        "work_parcel",
        "work_taz",
        "prev_work_parcel",
        "prev_work_taz",
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

    trip = locate_trip_parcels(trip_original.copy(), parcel_df, df_taz)

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

    # Write to file
    trip_original_updated.to_csv(
        os.path.join(config["output_dir"], "geolocated_trip.csv"), index=False
    )

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------locate_parcels.py ENDING--------------------")
    logger.info("locate_parcels.py RUN TIME %s" % str(elapsed_total))
