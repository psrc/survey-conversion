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

pd.options.mode.chained_assignment = None  # default='warn'


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


def locate_person_parcels(person, parcel_df, filter_dict_list, config):
    """Locate parcel ID for school, workplace, home location from person records."""

    person_results = person.copy()  # Make local copy for storing resulting joins

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
        gdf = gdf.merge(
            parcel_df[["parcel_id", "taz_id", "maz_id"]],
            left_on=varname + "_parcel",
            right_on="parcel_id",
        )
        gdf.rename(
            columns={"taz_id": varname + "_taz", "maz_id": varname + "_maz"},
            inplace=True,
        )
        gdf_cols = [
            "person_id",
            varname + "_taz",
            varname + "_maz",
            varname + "_parcel",
            varname + "_parcel_distance",
            varname + "_lat_fips_4601",
            varname + "_lng_fips_4601",
            varname + "_lat_gps",
        ]

        # Refine School Location in 2 tiers
        # Tier 2: for locations that are over 1 mile (5280 feet) from lat/lng,
        # place them in parcel with > 0 education or service employees (could be daycare or specialized school, etc. without students listed)

        if varname == "school_loc":
            gdf_far = gdf[gdf[varname + "_parcel_distance"] > config["school_max_dist"]]
            _dist, _ix = locate_parcel(
                parcel_df[parcel_df["total_students"] > 0],
                df=gdf_far,
                xcoord_col=varname + "_lng_fips_4601",
                ycoord_col=varname + "_lat_fips_4601",
            )
            gdf_far[varname + "_parcel"] = parcel_df.iloc[_ix].parcelid.values
            gdf_far[varname + "_parcel_distance"] = _dist

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


def locate_hh_parcels(hh, parcel_df, filter_dict_list, config):
    hh_results = hh.copy()

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
        gdf = gdf.merge(
            parcel_df[["parcel_id", "taz_id", "maz_id"]],
            left_on=varname + "_parcel",
            right_on="parcel_id",
        )
        gdf.rename(
            columns={"taz_id": varname + "_taz", "maz_id": varname + "_maz"},
            inplace=True,
        )

        # For households that are not reasonably near a parcel with households,
        # add them to the nearset unfiltered parcel and flag
        # Typically occurs with households living on military bases
        gdf_far = gdf[gdf[varname + "_parcel_distance"] > config["hh_max_dist"]]
        _dist, _ix = locate_parcel(
            parcel_df,
            df=gdf_far,
            xcoord_col=varname + "_lng_fips_4601",
            ycoord_col=varname + "_lat_fips_4601",
        )
        gdf_far[varname + "_parcel"] = parcel_df.iloc[_ix].parcelid.values
        gdf_far[varname + "_parcel_distance"] = _dist
        gdf_far = gdf_far.merge(
            parcel_df[["parcel_id", "taz_id", "maz_id"]],
            left_on=varname + "_parcel",
            right_on="parcel_id",
        )
        gdf_far.rename(
            columns={"taz_id": varname + "_taz", "maz_id": varname + "_maz"},
            inplace=True,
        )

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
                    varname + "_maz",
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


def locate_trip_parcels(
    trip, parcel_df, opurp_field, dpurp_field, filter_dict_list, config
):
    """Attach parcel ID to trip origins and destinations."""

    trip_results = trip.copy()

    for trip_end in ["origin", "dest"]:
        lng_field = trip_end + "_lng"
        lat_field = trip_end + "_lat"

        gdf = gpd.GeoDataFrame(
            trip, geometry=gpd.points_from_xy(trip[lng_field], trip[lat_field])
        )

        # convert from lat/lng to state plane WA
        gdf.crs = config["lat_lng_crs"]
        gdf = gdf.to_crs(config["wa_state_plane_crs"])

        xy_field = get_points_array(gdf.geometry)
        gdf[trip_end + "_lng_gps"] = gdf[trip_end + "_lng"]
        gdf[trip_end + "_lat_gps"] = gdf[trip_end + "_lat"]
        gdf[trip_end + "_lng_fips_4601"] = xy_field[:, 0]
        gdf[trip_end + "_lat_fips_4601"] = xy_field[:, 1]

        trip_results = trip_results.merge(
            gdf[
                [
                    trip_end + "_lng_gps",
                    trip_end + "_lat_gps",
                    trip_end + "_lng_fips_4601",
                    trip_end + "_lat_fips_4601",
                ]
            ],
            left_index=True,
            right_index=True,
        )

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
            _df["trip_id"] = _df.index
            _df = _df.merge(
                parcel_df[["parcel_id", "taz_id", "maz_id"]],
                left_on=trip_end_type[0] + "pcl",
                right_on="parcel_id",
            )
            _df.rename(
                columns={
                    "taz_id": trip_end_type[0] + "taz",
                    "maz_id": trip_end_type[0] + "maz",
                },
                inplace=True,
            )

            df_temp = df_temp.append(_df)
        # Join df_temp to final field for each trip type
        df_temp.set_index("trip_id", inplace=True)
        final_df = final_df.merge(
            df_temp[
                [
                    trip_end_type[0] + "pcl",
                    trip_end_type[0] + "pcl_distance",
                    trip_end_type[0] + "taz",
                    trip_end_type[0] + "maz",
                ]
            ],
            left_index=True,
            right_index=True,
            how="left",
        )

    return final_df
