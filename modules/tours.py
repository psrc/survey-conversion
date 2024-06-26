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
from modules import convert

pd.options.mode.chained_assignment = None  # default='warn'

# Keep track of error types
error_dict = {
    "no error": 0,
    "no home origin purpose": 1,
    "no home destinatation purpose": 2,
    "first O and last D are not home": 3,
    "dpurp of previous trip does not match opurp of next trip": 4,
    "activity type of previous trip does not match next trip": 5,
    "different number of tour starts/ends at home": 6,
    "no trips in set": 7,
    "no purposes provided except change mode": 8,
    "trip set includes start/end times in next day": 9
}

note_dict = {"changed date": 1}

def flag_issue(df, trip, error_str):

    df['error'] = error_str
    df['error_flag'] = error_dict[error_str]

    trip.loc[trip["trip_id"].isin(df["trip_id"].values), "error_flag"] = error_dict[error_str]

    return df, trip

def make_note(df, trip, note_str):

    trip.loc[trip["trip_id"].isin(df["trip_id"].values), "note_flag"] = note_dict[note_str]

    return trip

def build(df, tour_dict, tour_id, trip, day, config):
    """
    Process set of trips
    """

    # Identify home-based tours
    home_tours_start = df[(df["opurp"] == config["home_purp"]) & (df['error_flag']==0)]
    home_tours_end = df[(df["dpurp"] == config["home_purp"]) & (df['error_flag']==0)]

    ########################################
    # Process Each Home-Based Tour
    ########################################
    # Loop through each of the home-based tours identified above
    for tour_start_index in range(len(home_tours_start)):
        tour_dict[tour_id] = {}

        # Iterate between the first and last row of trips to build the tour
        start_row_id = home_tours_start.index[tour_start_index]
        end_row_id = home_tours_end.index[tour_start_index]

        # Select slice of trips that represent a set of home-to-home trips
        _df = df.loc[start_row_id:end_row_id]

        if len(_df) == 0:
            _df, trip = flag_issue(_df, trip, "no trips in set")
            continue

        # If no other trip purposes besides home and change mode, flag and skip
        if (
            _df.loc[_df["dpurp"] != config["home_purp"], "dpurp"].unique()[0]
            == config["change_mode_purp"]
        ):
            _df, trip = flag_issue(_df, trip, "no purposes provided except change mode")
            continue

        # Calculate duration at location between each trip
        # (difference between arrival at a destination and start of next trip)
        _df["duration"] = (
            _df.shift(-1).iloc[:-1]["deptm"] - _df.iloc[:-1]["arrtm"]
        )

        ########################################
        # Check for Subtours
        ########################################
        # Keep track of subtours for ID indexing
        subtour_count = 0

        # Check if we are dealing with work-based subtours
        # There must be 4+ trips for a subtour to exist; usual workplace destination appears at least twice (oadtyp==2)
        # Ensure we don't capture return trips home by requiring destination purpose of usual workplace and non-home types
        if (
            (len(_df) >= 4)
            & (len(_df[_df["oadtyp"] == 2]) >= 2)
            & (len(_df[_df["opurp"] == config['work_purp']]) >= 2)
            & (len(_df[(_df["oadtyp"] == 2) & (_df["dadtyp"] > 2)]) >= 1)
        ):
            # Potential that subtours exist; test each one to see if it's usable
            subtour_index_start_values = _df[
                (_df["oadtyp"] == 2) & (_df["dadtyp"] > 2)
            ].index.values
            subtours_df = pd.DataFrame()

            # Loop through each potential subtour
            # the following trips must eventually return to work for this to qualify as a subtour

            ########################################
            # Process Subtours
            ########################################
            for subtour_start_value in subtour_index_start_values:
                # Potential subtour
                # Loop through the index from subtour start
                # Create a new subtour ID; subtour_count is 0-based so add 1 to increment
                subtour_id = tour_id + subtour_count + 1

                next_row_index_start = (
                    np.where(_df.index.values == subtour_start_value)[0][0] + 1
                )
                for i in _df.index.values[next_row_index_start:]:
                    next_row = _df.loc[i]
                    # If next trip is to usual workplace, this marks the end of the subtour
                    if next_row["dadtyp"] == 2:
                        subtour_df = _df.loc[subtour_start_value:i]

                        tour_dict[subtour_id] = {}
                        # Process this subtour
                        # Create a new tour record for the subtour
                        subtour_df["subtour_id"] = subtour_id
                        subtours_df = pd.concat([subtours_df,subtour_df])

                        # Subtours are added as separate tour records
                        tour_dict[subtour_id]["tour"] = subtour_id

                        tour_dict = convert.add_tour_data(
                            subtour_df, tour_dict, subtour_id, day, config
                        )

                        tour_dict[subtour_id][
                            "parent"
                        ] = tour_id  # Parent is the main tour ID
                        tour_dict[subtour_id][
                            "subtrs"
                        ] = 0  # No subtours for subtours

                        trip = convert.update_trip_data(
                            trip, subtour_df, subtour_id
                        )

                        subtour_count += 1

                        break
                    else:
                        continue

            # No subtours actually found, treat as regular set of trips
            if len(subtours_df) < 1:
                tour_dict[tour_id]["subtrs"] = 0
                tour_dict[tour_id]["parent"] = 0

                tour_dict = convert.add_tour_data(
                    _df, tour_dict, tour_id, day, config
                )

                trip = convert.update_trip_data(trip, _df, tour_id)

            else:
                # Subtours were identified
                # Fill out primary tour and trip data
                for col in [
                    "hhno",
                    "pno",
                    "person_id",
                ]:
                    tour_dict[tour_id][col] = _df.iloc[0][col]

                tour_dict[tour_id]["day"] = day
                # First trip row contains departure time and origin info
                tour_dict[tour_id]["tlvorig"] = df.iloc[0]["deptm"]
                for col in ['opcl','omaz','otaz']:
                    if col in df.columns:
                        tour_dict[tour_id]["t"+col] = df.iloc[0][col]
                tour_dict[tour_id]["toadtyp"] = df.iloc[0]["oadtyp"]

                # Last trip row contains return info
                tour_dict[tour_id]["tarorig"] = df.iloc[-1]["arrtm"]

                # The main tour destination arrival will be the trip before subtours
                # the main tour destination departure will be the trip after subtours
                # trip when they arrive to work -> always the previous trip before subtours_df index begins
                main_tour_start_index = _df.index.values[
                    np.where(_df.index.values == subtours_df.index[0])[0][0] - 1
                ]
                # trip when leave work -> always the next trip after the end of the subtours_df
                main_tour_end_index = _df.index.values[
                    np.where(_df.index.values == subtours_df.index[-1])[0][0]
                    + 1
                ]
                # If there were subtours, this is a work tour
                tour_dict[tour_id]["pdpurp"] = config['work_purp']
                for col in ['dpcl','dmaz','dtaz']:
                    if col in df.columns:
                        tour_dict[tour_id]["t"+col] = _df.loc[main_tour_start_index][
                            col
                        ]
                tour_dict[tour_id]["tdadtyp"] = _df.loc[main_tour_start_index][
                    "dadtyp"
                ]

                if "pathtype" in df.columns:
                    # Pathtype is a heirarchy of modes, set in config in "mode_heirarchy"
                    # Note that tour pathtype is different from trip path type
                    subtours_excluded_df = pd.concat(
                        [
                            df.loc[start_row_id:main_tour_start_index],
                            df.loc[main_tour_end_index:end_row_id],
                        ]
                    )
                    tour_dict[tour_id]["tpathtp"] = subtours_excluded_df.loc[
                        subtours_excluded_df["mode"].idxmax()
                    ]["pathtype"]

                # Calculate tour halves, etc
                tour_dict[tour_id]["tripsh1"] = len(
                    _df.loc[0:main_tour_start_index]
                )
                tour_dict[tour_id]["tripsh2"] = len(
                    _df.loc[main_tour_end_index:]
                )

                # Set tour halves on trip records
                trip.loc[
                    trip["trip_id"].isin(
                        _df.loc[0:main_tour_start_index].trip_id
                    ),
                    "half",
                ] = 1
                trip.loc[
                    trip["trip_id"].isin(_df.loc[main_tour_end_index:].trip_id),
                    "half",
                ] = 2

                # set trip segment within half tours
                trip.loc[
                    trip["trip_id"].isin(
                        _df.loc[0:main_tour_start_index].trip_id
                    ),
                    "tseg",
                ] = range(1, len(_df.loc[0:main_tour_start_index]) + 1)
                trip.loc[
                    trip["trip_id"].isin(_df.loc[main_tour_end_index:].trip_id),
                    "tseg",
                ] = range(1, len(_df.loc[main_tour_end_index:]) + 1)

                # Departure/arrival times
                tour_dict[tour_id]["tlvdest"] = _df.loc[main_tour_end_index][
                    "deptm"
                ]
                tour_dict[tour_id]["tardest"] = _df.loc[main_tour_start_index][
                    "arrtm"
                ]

                # Number of subtours
                tour_dict[tour_id]["subtrs"] = subtour_count
                tour_dict[tour_id]["parent"] = 0

                # Mode
                tour_dict[tour_id]["tmodetp"] = convert.assign_tour_mode(
                    _df, config
                )

                tour_dict[tour_id]["tour"] = tour_id

                # add tour ID to the trip records (for trips not in the subtour_df)
                df_unique_no_subtours = [
                    i
                    for i in _df["trip_id"].values
                    if i not in subtours_df["trip_id"].values
                ]
                df_unique_no_subtours = _df[
                    _df["trip_id"].isin(df_unique_no_subtours)
                ]
                trip.loc[
                    trip["trip_id"].isin(
                        df_unique_no_subtours["trip_id"].values
                    ),
                    "tour",
                ] = tour_id

        else:
            # No subtours
            # Add tour data and update trip data
            tour_dict = convert.add_tour_data(
                _df, tour_dict, tour_id, day, config
            )
            # No subtours
            tour_dict[tour_id]["subtrs"] = 0
            tour_dict[tour_id]["parent"] = 0

            tour_dict[tour_id]["tour"] = tour_id

            trip = convert.update_trip_data(trip, _df, tour_id)

        tour_id = tour_id + subtour_count + 1

    return tour_dict, tour_id, trip

def create(trip, config):
    """
    Create tours from trip records.
    Flag errors in tour formation.
    """
    tour_dict = {}
    tour_id = 1

    trip['error_flag'] = 0

    if len(config["debug_person_id"]) > 0:
        person_id_list = config["debug_person_id"]
    else:
        # Get all unique person ID values
        person_id_list = trip["person_id"].value_counts().index.values

    for person_id in person_id_list:
        print(str(tour_id))
        person_df = trip.loc[trip["person_id"] == person_id]

        # Evaluate each person's travel day
        for day in person_df["day"].unique():
            df = person_df.loc[person_df["day"] == day]

            df = df.reset_index(drop=True)

            # Travel days should start and end at 3 am so we can capture as much activity as possible
            # We need to make this adjustment at the day level since trips can begin on a previous day
            # or end on the next day (or both)
            # To do this we need to create new 3am travel dates

            # for trips occuring between midnight and 3 am,
            # change the travel date of all trips to match the first trip
            if len(df[df['deptm'] < (3*60)]) > 0:
                for col in ['depart_date', 'arrive_date']:
                    df[col] = df.iloc[0]['depart_date']
                    trip = make_note(df[df['deptm'] < (3*60)], trip, "changed date")

            # Exclude travel across multiple days (past 3 AM)
            if len(df['depart_date'].unique()) > 1 or len(df['arrive_date'].unique()) > 1:
                df, trip = flag_issue(df, trip, "trip set includes start/end times in next day")
                # Assume first set of trips represent travel day; flag others
                primary_date = df.iloc[0]['depart_date']
                filter = ((df['depart_date'] == primary_date) & 
                         (df['arrive_date'] == primary_date))
                df, trip = flag_issue(df[filter], trip, "no error")
            
            # If home origin not available for any trip in set, flag and continue on to another set
            if not (df['opurp'] == config['home_purp']).any():
                df, trip = flag_issue(df, trip, "no home origin purpose")
                continue
            if not (df['dpurp'] == config['home_purp']).any():
                df, trip = flag_issue(df, trip, "no home destinatation purpose")
                continue

            # If first origin and last destination are not home, flag and remove first/last trips until criteria is met
            if (
                df.groupby("person_id").first()["opurp"].values[0]
                != config["home_purp"]
            ) or df.groupby("person_id").last()["dpurp"].values[0] != config[
                "home_purp"
            ]:
                df, trip = flag_issue(df, trip, "first O and last D are not home")
                
                # Work forwards using i-index from beginning until home origin purpose is found   
                i = 0
                while df.iloc[i:].groupby("person_id").first()["opurp"].values[0] != config["home_purp"]:
                    i += 1

                # Work backwards using j-index from end of trip list until home destination purpose is found
                j = len(df)-1
                while df.loc[j:].groupby("person_id").first()["dpurp"].values[0] != config["home_purp"]:
                    j -= 1
                # Once home destination purpose is found, remove flag for remaining records between i and j indeces 
                df, trip = flag_issue(df.loc[i:j], trip, "no error")

            # Check that sequential origin and destination purposes match
            df["next_opurp"] = df.shift(-1)[["opurp"]]
            df["prev_opurp"] = df.shift(1)[["opurp"]]
            df["next_oadtyp"] = df.shift(-1)[["oadtyp"]]
            if (
                len(df.iloc[:-1][(df.iloc[:-1]["next_opurp"] != df.iloc[:-1]["dpurp"])])
                > 0
            ):
                df, trip = flag_issue(df, trip, "dpurp of previous trip does not match opurp of next trip")

                # FIXME: try to stitch these together if needed

            # Apply similarly for activity type
            if (
                len(
                    df.iloc[:-1][
                        (df.iloc[:-1]["next_oadtyp"] != df.iloc[:-1]["dadtyp"])
                    ]
                )
                > 0
            ):
                df, trip = flag_issue(df, trip, "activity type of previous trip does not match next trip")    

            # Continue working with the set of filtered trips only
            df = df[df['error_flag'] == 0]       

            tour_dict, tour_id, trip = build(df, tour_dict, tour_id, trip, day, config)

    tour = pd.DataFrame.from_dict(tour_dict, orient="index")

    return tour, trip
