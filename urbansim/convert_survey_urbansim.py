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
from daysim import logcontroller, convert_format
from modules import util, convert, tours, days

pd.options.mode.chained_assignment = None  # default='warn'


def convert_format_urbansim(config):
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

    # Load geolocated survey data
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
        df_lookup[df_lookup["table"] == "person"],
    )

    hh = convert_format.process_household_file(
        hh_original_df, person, df_lookup, config, logger
    )

    for df_name, df in {
        "1_household": hh,
        "2_person": person,
    }.items():
        print(df_name)

        df.to_csv(os.path.join(config["output_dir"], df_name + ".csv"), index=False)

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------convert_format.py ENDING--------------------")
    logger.info("convert_format.py RUN TIME %s" % str(elapsed_total))
