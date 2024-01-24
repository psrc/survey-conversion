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

config = toml.load("daysim_configuration.toml")

def apply_filter(df, df_name, filter, logger, msg):
    """Apply a filter to trim df, record changes in log file."""

    logger.info(f"Dropped {len(df[~filter])} " + df_name + ": " + msg)

    return df[filter]

def assign_tour_mode(_df, tour_dict, tour_id, mode_heirarchy=config["mode_heirarchy"]):
    """Get a list of transit modes and identify primary mode
    Primary mode is the first one from a heirarchy list found in the tour."""
    mode_list = _df["mode"].value_counts().index.astype("int").values

    for mode in mode_heirarchy:
        if mode in mode_list:
            # If transit, check whether access mode is walk to transit or drive to transit
            if mode == 6:
                # Try to use the access mode field values to get access mode
                try:
                    if len([i for i in _df["mode_acc"].values if i in [3, 4, 5, 6, 7]]):
                        tour_mode = 7  # park and ride
                    else:
                        tour_mode = 6
                    return tour_mode
                except:
                    # otherwise, assume walk to transit
                    tour_mode = 6  # walk
                    return tour_mode

            return mode
        

def process_expression_file(df, expr_df, output_column_list):
    """Execute each row of calculations in an expression file. 
    Fill empty columns in output_column_list with -1."""

    for index, row in expr_df.iterrows():
        expr = (
            "df.loc["
            + row["filter"]
            + ', "'
            + row["result_col"]
            + '"] = '
            + str(row["result_value"])
        )
        print(row["index"])
        exec(expr)

    # Add empty columns to fill in later with skims
    for col in output_column_list:
        if col not in df.columns:
            df[col] = -1
        else:
            df[col] = df[col].fillna(-1)

    df = df[output_column_list]

    return df