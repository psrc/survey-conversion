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

import pandas as pd
import numpy as np
import os
import toml
import pandera as pa
from pandera import Column, Check

tours_schema = pa.DataFrameSchema(
    {
        "tour_id": Column(int, nullable=False),
        "person_id": Column(int, nullable=False),
        "household_id": Column(int, nullable=False),
        "tour_type": Column(str, nullable=False),
        "tour_category": Column(str, nullable=False),
        "destination": Column(int, nullable=False),
        "origin": Column(int, nullable=False),
        "start": Column(int, nullable=False),
        "end": Column(int, nullable=False),
        "tour_mode": Column(str, nullable=False),
        "parent_tour_id": Column(int, nullable=False),
        "household_id_elmer": Column(int, nullable=False),
    },
    coerce=True,
)

trips_schema = pa.DataFrameSchema(
    {
        "trip_id": Column(int, nullable=False),
        "tsvid": Column(int, nullable=False),
        "person_id": Column(int, nullable=False),
        "PNUM": Column(int, nullable=False),
        "household_id": Column(int, nullable=False),
        "day": Column(int, nullable=False),
        "outbound": Column(str, nullable=False),
        "purpose": Column(int, nullable=False),
        "opurp": Column(str, nullable=False),
        "purpose": Column(str, nullable=False),
        "oadtyp": Column(int, Check.isin([1, 2, 3, 4, 5, 6]), nullable=False),
        "dadtyp": Column(int, Check.isin([1, 2, 3, 4, 5, 6]), nullable=False),
        "origin": Column(int, nullable=False),
        "destination": Column(int, nullable=False),
        "depart": Column(int, nullable=False),
        "arrtm": Column(int, nullable=False),
        "trip_mode": Column(str, nullable=False),
        "trip_weight": Column(int, nullable=False),
        "tour_id": Column(float, nullable=False),
        "trip_num": Column(float, nullable=False),
    },
    coerce=True,
)

household_schema = pa.DataFrameSchema(
    {
        "household_id": Column(int, nullable=False),
        "home_zone_id": Column(int, nullable=False),
        "income": Column(int, nullable=False),
        "hhsize": Column(int, nullable=False),
        "HHT": Column(int, nullable=False),
        "auto_ownership": Column(int, nullable=False),
        "num_workers": Column(int, nullable=False),
        "hh_race_category": Column(str, nullable=False),
        "hh_weight": Column(int, nullable=False),
        "household_id_original": Column(int, nullable=False),
    },
    coerce=True,
)

person_schema = pa.DataFrameSchema(
    {
        "person_id": Column(int, nullable=False),
        "household_id": Column(int, nullable=False),
        "age": Column(int, nullable=False),
        "PNUM": Column(int, nullable=False),
        "sex": Column(int, nullable=False),
        "pemploy": Column(int, nullable=False),
        "pstudent": Column(int, nullable=False),
        "ptype": Column(int, nullable=False),
        "school_zone_id": Column(int, nullable=False),
        "workplace_zone_id": Column(int, nullable=False),
        "free_parking_at_work": Column(int, nullable=False),
        "household_id_original": Column(int, nullable=False),
        "person_id_elmer_original": Column(int, nullable=False),
        "person_weight": Column(float, nullable=False),
        "telecommute_frequency": Column(str, nullable=False),
    },
    coerce=True,
)


def read_validate_write(schema, fname):
    """Load survey file, apply schema, and overwrite results."""

    df = pd.read_csv(fname)
    df = schema.validate(df.fillna(-1))
    df[schema.columns.keys()].to_csv(fname, index=False)


def data_validation(config):
    read_validate_write(tours_schema, 
                        os.path.join(config["output_dir"],'cleaned', "survey_tours.csv"))
    read_validate_write(trips_schema, 
                        os.path.join(config["output_dir"],'cleaned', "survey_trips.csv"))
    read_validate_write(person_schema, 
                        os.path.join(config["output_dir"],'cleaned', "survey_persons.csv"))
    read_validate_write(household_schema, 
                        os.path.join(config["output_dir"],'cleaned', "survey_households.csv"))
    