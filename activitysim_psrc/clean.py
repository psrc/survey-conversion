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
from daysim import logcontroller
from activitysim_psrc import infer
from modules import convert
from activitysim.core import workflow
from activitysim.abm.models.util import canonical_ids as cid
from activitysim import cli as client
import sys
import argparse

survey_tables = {
    "households": {"file_name": "survey_households.csv"},
    "persons": {"file_name": "survey_persons.csv"},
    "tours": {"file_name": "survey_tours.csv"},
    "joint_tour_participants": {"file_name": "survey_joint_tour_participants.csv"},
    "trips": {"file_name": "survey_trips.csv"},
}

def remove_hh_with_missing_work_location(household, person, tours, trips, joint_tour_participants, logger):
    remove_persons = person[(person['pemploy'].isin([1,2])) & (person['workplace_zone_id']==-1)].person_id
    remove_hh = person[person['person_id'].isin(remove_persons)].household_id.unique()
    
    person = person[~person['person_id'].isin(remove_persons)]
    household = household[~household['household_id'].isin(remove_hh)]
    
    logger.info(
        f"Dropped {len(remove_hh)} households: missing work locations"
    )

    trips = trips[~trips['household_id'].isin(remove_hh)]
    tours = tours[~tours['household_id'].isin(remove_hh)]

    joint_tour_participants = joint_tour_participants[~joint_tour_participants['household_id'].isin(remove_hh)]

    return household, person, tours, trips, joint_tour_participants

def recode_missing_school_location_to_home(household, person, tour, trip):
    df = person.merge(household[['household_id', 'home_zone_id']], how ='left', on='household_id')
    df['school_zone_id'] = np.where((df.pstudent<3) & (df.school_zone_id == -1), df.home_zone_id, df.school_zone_id)
    
    return df[person.columns]

def process_person_day(tour, trip, config, logger, state):
    # In order to estimate, we need to enforce the mandatory tour totals
    # these can only be: ['work_and_school', 'school1', 'work1', 'school2', 'work2']
    # If someone has 2 work trips and 1 school trips, must decide a heirarchy of
    # which of those tours to delete

    # FIXME: how do we handle people with too many mandatory tours?
    # Do we completely ignore all of this person’s tours, select the first tours,
    # or use some other logic to identify the primary set of tours and combinations?

    person_day = tour.groupby("person_id").agg(["unique"])["loc_tour_id"]

    person_day["flag"] = 0

    # Flag any trips that have 3 or more work or school tours

    # Flag 1: person days that have 2 work and 2 school tours
    filter = person_day["unique"].apply(lambda x: "work2" in x and "school2" in x)
    person_day.loc[filter, "flag"] = 1
    # Resolve by: dropping all work2 and school2 tours (?) FIXME...
    tour = tour[
        ~(
            (tour["person_id"].isin(person_day[person_day["flag"] == 1].index))
            & tour["tour_id"].isin(["work2", "school2"])
        )
    ]

    # Flag 2: 2 work tours and 1 school tour
    filter = person_day["unique"].apply(lambda x: "work2" in x and "school1" in x)
    person_day.loc[filter, "flag"] = 2
    # Resolve by: dropping all work2 tours  (?) FIXME...
    tour = tour[
        ~(
            (tour["person_id"].isin(person_day[person_day["flag"] == 2].index))
            & (tour["tour_id"] == "work2")
        )
    ]

    # Flag 3: 2 school tours and 1 work tour
    filter = person_day["unique"].apply(lambda x: "work1" in x and "school2" in x)
    person_day.loc[filter, "flag"] = 3
    # Resolve by: dropping all school2 tours (?) FIXME...
    tour = tour[
        ~(
            (tour["person_id"].isin(person_day[person_day["flag"] == 3].index))
            & (tour["tour_id"] == "school2")
        )
    ]

    # Report number of tours affected
    # FIXME: write out a log file
    # print(str(person_day.groupby("flag").count()))

    # DATA FILTER:
    # Default of 4, determine based on config files
    # sys.argv.append('--working_dir')
    # sys.argv.append(config['asim_config_dir'])
    # parser = argparse.ArgumentParser()
    # client.run.add_run_args(parser)
    # args = parser.parse_args()
    # state = workflow.State()
    # state.logging.config_logger(basic=True)
    # state = client.run.handle_standard_args(state, args)  # possibly update injectables


    possible_tours = cid.canonical_tours(state)
    MAX_TRIPS_PER_LEG = cid.determine_max_trips_per_leg(state)
    # Filter out tours with too many stops on their tours
    df = tour[(tour["tripsh1"] > MAX_TRIPS_PER_LEG) | (tour["tripsh2"] > MAX_TRIPS_PER_LEG)]
    # df.to_csv(os.path.join(output_dir,'temp','too_many_stops.csv'))
    logger.info(f"Dropped {len(df)} tours for too many stops")
    tour = tour[~((tour["tripsh1"] > MAX_TRIPS_PER_LEG) | (tour["tripsh2"] > MAX_TRIPS_PER_LEG))]

    # DATA FILTER:
    # select trips that only exist in tours - is this necessary or can we use the trip file directly?

    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    # Keep the original trip ID for later use
    canonical_trip_num = (~trip.outbound * MAX_TRIPS_PER_LEG) + trip.trip_num
    trip["trip_id"] = trip["tour"] * (2 * MAX_TRIPS_PER_LEG) + canonical_trip_num

    # DATA FILTER:
    # Some of these IDs are duplicated and it's not clear why - seems to be an issue with the canonical_trip_num definition
    # FIXME: what do we do about this? Fix canonical_trip_num? drop duplicates?
    duplicated_person = trip[trip["trip_id"].duplicated()]["person_id"].unique()
    logger.info(
        f"Dropped {len(duplicated_person)} persons: duplicate IDs from canonical trip num definition"
    )
    trip = trip[~trip["person_id"].isin(duplicated_person)]
    trip.set_index("trip_id", inplace=True, drop=False, verify_integrity=True)

    # Make sure all trips in a tour have an outbound and inbound component
    trips_per_tour = trip.groupby("tour")["person_id"].value_counts()
    missing_trip_persons = (
        trips_per_tour[trips_per_tour == 1]
        .index.get_level_values("person_id")
        .to_list()
    )
    logger.info(
        f" {len(missing_trip_persons)} persons missing an outbound or inbound trip leg"
    )

    return tour, trip

def clean(config, state):
    # Start log file
    logger = logcontroller.setup_custom_logger("convert_format_logger.txt", config)
    logger.info("--------------------convert_format.py STARTING--------------------")
    start_time = datetime.datetime.now()

    households = pd.read_csv(os.path.join(config['output_dir'], 'survey_households.csv')) 
    person = pd.read_csv(os.path.join(config['output_dir'], 'survey_persons.csv')) 
    tour = pd.read_csv(os.path.join(config['output_dir'], 'survey_tours.csv')) 
    joint_tour_participants = pd.read_csv(os.path.join(config['output_dir'], 'survey_joint_tour_participants.csv')) 
    trip = pd.read_csv(os.path.join(config['output_dir'], 'survey_trips.csv')) 

    # Create new output directory for survey records with skims attached, if needed
    cleaned_output_dir = os.path.join(config["output_dir"], "cleaned")
    
    for my_path in [cleaned_output_dir]:
        if not os.path.exists(my_path):
            os.makedirs(my_path)

    # Remove trips with tour flags
    filter = tour['error_str'].notnull()
    logger.info(f"Dropped {len(tour[filter])} tours: non-canonical tour type")
    tour = tour[~filter]

    # person day
    tour, trip = process_person_day(tour, trip, config, logger, state)

    # Process joint tours


    # Map to standard columns
    df_mapping = pd.read_csv(os.path.join(os.getcwd(), config["input_dir"], "mapping.csv"))

    tour.drop("tour_id", inplace=True, axis=1)
    tour = convert.map_to_class(tour, "tour", df_mapping, "script_to_original")
    trip = convert.map_to_class(trip, "trip", df_mapping, "script_to_original")
    person = convert.map_to_class(person, "person", df_mapping, "script_to_original")
    joint_tour_participants = convert.map_to_class(
        joint_tour_participants, "joint_tour", df_mapping, "script_to_original"
    )


    # Filter trip data
    for col in ["origin", "destination"]:
        # Missing omaz/dmaz
        trip.loc[trip[col] <= 0, 'error_flag'] = 10
        trip.loc[trip[col] <= 0, 'error_flag'] = 10


    filter = ((trip["opurp"] == trip["purpose"]) & (trip["opurp"] == "Home")),
    trip.loc[trip[col] <= 0, 'error_flag'] = 11 # trips have same origin/destination of home

    # Do we need to drop these trips/tours?

    # Drop any duplicate tour IDs
    _filter = tour["tour_id"].duplicated()
    logger.info(f"Dropped {len(tour[_filter])} tours: duplicate tour ID")
    tour = tour[~_filter]

    #########################################
    # at work tours
    #########################################

    # People can't make more than 1 eat or maint subtour,
    # or more than 2 business subtours on at-work tours
    # Keeping only the first and dropping the others
    atwork_tours = tour[tour["tour_category"] == "atwork"]
    atwork_tours = atwork_tours.groupby(["parent_tour_id", "tour_type"]).count()[
        ["household_id"]
    ]
    tour["drop"] = 0
    for tour_id in atwork_tours.index:
        df = atwork_tours.loc[tour_id[0]]
        if df["household_id"].sum() > 1:
            # Only a certain set of atwork stops are allowed
            # If 2 business stops, nothing else
            df = tour.loc[
                tour["parent_tour_id"] == tour_id[0], "tour_type"
            ].value_counts()
            if "business" in df.index:
                if df["business"] == 2:
                    # Take only first 2 business subtours and drop all else
                    tour.loc[
                        (tour["parent_tour_id"] == tour_id[0])
                        & (tour["tour_type"] != "business"),
                        "drop",
                    ] = 1
                if "eat" in df.index:
                    # Take first business and first eat subtour only; drop any others
                    if df["eat"] > 1:
                        tour.loc[
                            (tour["parent_tour_id"] == tour_id[0])
                            & (tour["tour_type"] == "eat"),
                            "drop",
                        ] = range(df["eat"].sum())
                    if df["business"] > 1:
                        tour.loc[
                            (tour["parent_tour_id"] == tour_id[0])
                            & (tour["tour_type"] == "business"),
                            "drop",
                        ] = range(df["business"].sum())
                if "eat" not in df.index and df["business"] < 2:
                    # Drop any other subtours
                    tour.loc[
                        (tour["parent_tour_id"] == tour_id[0])
                        & (tour["tour_type"] != "business"),
                        "drop",
                    ] = 1
            if "business" not in df.index:
                # Take first subtour only
                tour.loc[(tour["parent_tour_id"] == tour_id[0]), "drop"] = range(
                    df.sum()
                )

    logger.info(
        f'Dropped {len(tour[tour["drop"] > 0])} tours: at work subtour with too many eat or maint trips'
    )
    tour = tour[tour["drop"] == 0]

    # at work trips should have a purpose of at-work    
    at_work_tours_ids = at_work = tour[tour['tour_category']=='atwork'].tour_id
    trip.loc[trip["tour_id"].isin(at_work_tours_ids), "purpose",] = 'atwork' 


    ###############################
    # School and workplace cleaning
    ###############################

    # if person makes a school tour but doesn't have a usual school location, use the first school tour destination
    # FIXME: this may be an issue if there are no students/education employment; may need to snap these trips and locations again
    school_tours = tour[tour.tour_type == "school"]
    school_tours = school_tours.groupby("person_id").first()[["destination"]]
    person["school_zone_id"] = person["school_zone_id"].fillna(-1)
    person["school_zone_id"] = person["school_zone_id"].replace(0, -1)
    person = person.merge(
        school_tours, how="left", left_on="person_id", right_index=True
    )
    person.rename(columns={"destination": "school_dest"}, inplace=True)
    person.loc[
        (~person["school_dest"].isnull())
        & (
            (person["school_zone_id"] == config["missing_school_zone"])
            | (person["school_zone_id"].isnull())
        ),
        "school_zone_id",
    ] = person["school_dest"]
    person.drop("school_dest", axis=1, inplace=True)

    # Check that anyone coded as person tpye (ptype) of college student has a school_zone (imputed or stated)
    # If not, change their person type

    # Apply same rule for usual work location
    work_tours = tour[tour.tour_type == "work"]
    work_tours = work_tours.groupby("person_id").first()[["destination"]]
    person["workplace_zone_id"] = person["workplace_zone_id"].fillna(-1)
    person["workplace_zone_id"] = person["workplace_zone_id"].replace(0, -1)
    person = person.merge(work_tours, how="left", left_on="person_id", right_index=True)
    person.rename(columns={"destination": "work_dest"}, inplace=True)
    # where they have a work destination, but no usual workplace_zone, set to usual workplace_zone
    person.loc[
        (~person["work_dest"].isnull()) & (person["workplace_zone_id"] == -1),
        "workplace_zone_id",
    ] = person["work_dest"]
    person.drop("work_dest", axis=1, inplace=True)



    # If a person makes a school tour, make sure their pstudent and ptype is correct
    # Addressed bug in mandatory tour frequency model
    school_tours = tour[tour.tour_type == "school"]
    person.loc[
        person["person_id"].isin(school_tours["person_id"])
        & (person["pstudent"] == 3)
        & (person["age"] <= 18),
        "pstudent",
    ] = 1  # k12 student
    person.loc[
        person["person_id"].isin(school_tours["person_id"])
        & (person["pstudent"] == 3)
        & (person["age"] > 18),
        "pstudent",
    ] = 2  # college student

    # People coded as non-workers making work tours
    # Stefan- seems like this should check for age 18+
    work_tours = tour[tour.tour_type == "work"]
    person.loc[
        person["person_id"].isin(work_tours["person_id"]) & (person["pemploy"] == 3),
        "pemploy",
    ] = 2  # assume part-time workers

    # Some young children are coded as having work tours (likely going with parent)
    # Remove any work tours for someone coded as ptype 8
    tour = tour.merge(person[["person_id", "ptype", "work_from_home"]], on="person_id", how="left")
    filter = (tour["tour_type"] == "work") & (tour["ptype"] == 8)
    logger.info(f"Dropped {len(tour[filter])} tours: ptype 8 making work tours")
    tour = tour[~filter]

    # Work at home workers cannot make work tours, change to other maintenance:

    filter = (tour["tour_type"] == "work") & (tour["work_from_home"] == 1)
    remove_tour_ids = tour[filter].tour_id
    tour = tour[~filter]
    trip = trip[~trip.tour_id.isin(remove_tour_ids)]

    # filter_tour_ids = tour.loc[filter].tour_id
    # tour.loc[filter, "tour_type"] = 'othmaint'
    # tour.loc[filter, "pdpurp"] = 'othmaint'
    # tour.loc[filter, "tour_category"] = 'non_mandatory'
    # assert tour[filter].tour_type.unique()[0] == 'othmaint'
    # assert len(tour[filter].tour_type.unique()) == 1
    # assert tour[filter].tour_category.unique()[0] == 'non_mandatory'
    # assert len(tour[filter].tour_category.unique()) == 1

    # # now Trips
    # trip.loc[trip['tour_id'].isin(filter_tour_ids) & (trip.opurp=='work'), 'opurp'] = 'othmaint'
    # trip.loc[trip['tour_id'].isin(filter_tour_ids) & (trip.purpose=='work'), 'purpose'] = 'othmaint'

    # # work tours/trips must go to usual workplace location, change to other maintenance
    # # first need to remove sub tours/trips that do not originate from usual workplace location.
    at_work = tour[tour['tour_category']=='atwork']
    at_work = at_work.merge(person[['person_id', 'workplace_zone_id']], on='person_id', how='left') 
    filter_tour_ids = at_work[at_work['origin'] != at_work['workplace_zone_id']].tour_id
    # # remove from tours, trips
    tour = tour[~tour.tour_id.isin(filter_tour_ids)]
    trip = trip[~trip.tour_id.isin(filter_tour_ids)]
    
    # # now recode tours and trips taht do not to to usual workplace location:
    tour = tour.merge(person[['person_id', 'workplace_zone_id', 'school_zone_id']], on='person_id', how='left') 
    filter = (tour.pdpurp=='work') & (tour.destination != tour.workplace_zone_id)
    filter_tour_ids = tour.loc[filter].tour_id
    tour = tour[~tour.tour_id.isin(filter_tour_ids)]
    trip = trip[~trip.tour_id.isin(filter_tour_ids)]
    # tour.loc[filter, "tour_type"] = 'othmaint'
    # tour.loc[filter, "pdpurp"] = 'othmaint'
    # tour.loc[filter, "tour_category"] = 'non_mandatory'
    # # now Trips
    # trip.loc[trip['tour_id'].isin(filter_tour_ids) & (trip.opurp=='work'), 'opurp'] = 'othmaint'
    # trip.loc[trip['tour_id'].isin(filter_tour_ids) & (trip.purpose=='work'), 'purpose'] = 'othmaint'
    
    # school tours/trips must go to usual school location, change to other maintenance
    #tour = tour.merge(person[['person_id', 'workplace_zone_id']], on='person_id', how='left') 
    filter = (tour.pdpurp=='school') & (tour.destination != tour.school_zone_id)
    filter_tour_ids = tour.loc[filter].tour_id
    tour = tour[~tour.tour_id.isin(filter_tour_ids)]
    trip = trip[~trip.tour_id.isin(filter_tour_ids)]


    # Work tours/trips
    #assert tour[filter].tour_type.unique

    person.loc[
        person["person_id"].isin(school_tours["person_id"])
        & (person["pstudent"] == 3)
        & (person["age"] > 18),
        "pstudent",
    ] = 2  # college student

    logger.info(f"Dropped {len(tour[filter])} tours: work at home workers making work tours")
    tour = tour[~filter]


    # FIXME: move this to expression files ###
    # If a person has a usual workplace zone make them a part time worker (?) or remove their usual workplace location...
    person[
        (person["workplace_zone_id"] > 0)
        & (person["pemploy"] >= 3)
        & (person["age"] >= 16)
    ]["pemploy"] = 2

    

    ### We cannot have more than 2 joint tours per household. If so, make sure we remove those households/tours
    ### FIXME: should we remove the households or edit the tours so they are not joint, or otherwise edit them?
    joint_tours = tour[tour["tour_category"] == "joint"]
    _df = joint_tours.groupby("household_id").count()["tour_id"]
    too_many_jt_hh = _df[_df > 2].index

    # FIXME: For now remove all households; 
    # We should figure out how to better deal with these
    filter = tour["household_id"].isin(too_many_jt_hh)
    logger.info(f"Dropped {len(tour[filter])} tours: too many joint tours per household")
    tour = tour[~filter]


    # person_cols = ['person_id','household_id','age','PNUM','sex','pemploy','pstudent','ptype','school_zone_id','workplace_zone_id','free_parking_at_work', 'race_category', 'person_id_elmer', person_weight]
    # tour_cols = ['tour_id','person_id','household_id','tour_type','tour_category','destination','origin','start','end','tour_mode','parent_tour_id']
    # trip_cols = ['trip_id','person_id','household_id','tour_id','outbound','purpose','destination','origin','depart','trip_mode', 'trip_id_elmer', trip_weight]
    # hh_cols = ['household_id','home_zone_id','income','hhsize','HHT','auto_ownership','num_workers', 'hh_race_category', 'household_id_elmer', hh_weight]

    # Make sure all records align with available and existing households/persons
    _filter = person["household_id"].isin(households["household_id"])
    logger.info(f"Dropped {len(person[~_filter])} persons: missing household records")
    person = person[_filter]

    # All persons and household records must have trips/tours (?)
    # Note: removing PNUM==1 causes problems in joint tour frequency model
    # Possible fix: renumerate PNUM to start at 1. Make sure it's updated on any related trip/tour files

    # Make sure ptype matches employment
    person.loc[
        (person["pemploy"] == 3)
        & (person["ptype"].isin([1, 2]))
        & (person["age"] <= 18),
        "ptype",
    ] = 6
    person.loc[
        (person["pemploy"] == 3)
        & (person["ptype"].isin([1, 2]))
        & (person["age"] > 18)
        & (person["age"] <= 65),
        "ptype",
    ] = 4  # non-working adult
    person.loc[
        (person["pemploy"] == 3)
        & (person["ptype"].isin([1, 2]))
        & (person["age"] > 65),
        "ptype",
    ] = 5  # retired

    # Ensure valid trips and tours
    tour = tour[tour["origin"] > 0]
    tour = tour[tour["destination"] > 0]
    trip = trip[trip["origin"] > 0]
    trip = trip[trip["destination"] > 0]

    # Make sure all tours have a tour_type defined
    # FIXME: add to expression files
    _filter = tour["tour_type"] == "-1"
    logger.info(f"Dropped {len(tour[_filter])} tours: missing tour type")
    tour = tour[~_filter]

    # Check that tours have a valid mode
    # FIXME: should we try to impute a mode?
    _filter = tour["tour_mode"].isnull()
    logger.info(f"Dropped {len(tour[_filter])} tours: missing tour mode")
    tour = tour[~_filter]

    # Drop any tours and trips if there is only 1 trips; this might have caused by cleaning above
    trips_per_tour = trip.groupby("tour_id").count()[["person_id"]]
    remove_tour_list = trips_per_tour[trips_per_tour.person_id < 2].index.values
    trip = trip[~trip["tour_id"].isin(remove_tour_list)]

    # Make sure each tour has an inbound and outbound component
    outbound_tours = trip[trip["outbound"] == True].tour_id.unique()
    inbound_tours = trip[trip["outbound"] == False].tour_id.unique()
    _filter = ((tour.tour_id.isin(outbound_tours))) & (
        (tour.tour_id.isin(inbound_tours))
    )
    logger.info(
        f"Dropped {len(tour[~_filter])} tours: missing inbound or outbound trip component"
    )
    tour = tour[_filter]

    # Make sure sequence of departure times is sequential to apply trip_num; remove if not
    # This comes from trips that extend past midnight;
    # FIXME: this should go until 3 am
    drop_tours = []
    for tour_id in tour["tour_id"]:
        df = trip[trip["tour_id"] == tour_id]["depart"]
        if not df.is_monotonic_increasing:
            drop_tours.append(tour_id)
    _filter = tour["tour_id"].isin(drop_tours)
    logger.info(
        f"Dropped {len(tour[_filter])} tours: travel day passes beyond midnight"
    )
    tour = tour[~_filter]

    # Make sure tours and trips align; if trips were dropped the tour file should be updated to reflect changes
    # expect outbound last trip destination to be same as half tour destination
    trip_dest = (
        trip[trip["outbound"] == True]
        .groupby("tour_id")
        .last()[["destination"]]
        .reset_index()
    )
    tour.drop("destination", inplace=True, axis=1)
    tour = tour.merge(trip_dest, on="tour_id", how="left")

    # Check that home-based tours start at home
    # FIXME: consider asserting trip origin at home location? Trace the cause of this
    # FIXME: There are some snapping issues where origins might be 1 zone away from home location
    # We might need to do some additional cleaning so that if a trip origin/destination is
    # for home origin/purpose and home is within a half mile to change otaz to be home taz
    tour = tour.merge(
        households[["household_id", "home_zone_id"]], on="household_id", how="left"
    )
    _filter = (
        (tour["tour_category"] != "atwork") & (tour["origin"] == tour["home_zone_id"])
    ) | (tour["tour_category"] == "atwork")
    logger.info(
        f"Dropped {len(tour[~_filter])} tours: non-subtour tours must start at home zone"
    )
    tour = tour[_filter]

    # FIXME: Check that subtours start at work place

    # Explicitly-defined non workers should not be making work trips
    # Recode ptype to be part-time workers
    work_tours = tour[tour.tour_type == "work"]
    person.loc[
        (person["ptype"].isin([4])) & person["person_id"].isin(work_tours["person_id"]),
        "ptype",
    ] = 2

    # Make sure trip origins and destinations are sequential
    # Since we had to remove certain trips this can affect the tour
    # FIXME: we may want to repair these Os and Ds, but they may not all make sense if missing a trip
    # Remove all trips and tours that don't have a matching half tour destinations
    # FIXME: put this and above in the same loop...
    flag_tours = []
    for tour_id in tour["tour_id"].unique():
        df = trip[trip["tour_id"] == tour_id]
        # expect outbound last trip destination to be same as outbound first trip origin
        if (df.loc[df["outbound"] == True, "destination"].iloc[-1]) != (
            df.loc[df["outbound"] == False, "origin"].iloc[0]
        ):
            flag_tours = np.append(flag_tours, tour_id)
        # expect inbound first trip origin to be same as half tour destination
        if (df.loc[df["outbound"] == False, "origin"].iloc[0]) != (
            tour[tour.tour_id == tour_id]["destination"].values[0]
        ):
            flag_tours = np.append(flag_tours, tour_id)
        # expect inbound last trip destination to be same as half tour origin
        if (df.loc[df["outbound"] == False, "destination"].iloc[-1]) != (
            tour[tour.tour_id == tour_id]["origin"].values[0]
        ):
            flag_tours = np.append(flag_tours, tour_id)

    # Drop all flagged tours
    logger.info(
        f"Dropped {len(flag_tours)} tours: first half tour destination does not match last half tour origin"
    )
    tour = tour[~tour.tour_id.isin(flag_tours)]

    # If a university student makes a school trip, rename purpose from "school" to "univ"
    trip = trip.merge(
        person[["person_id", "pstudent", "ptype"]], on="person_id", how="left"
    )
    trip.loc[
        (trip["pstudent"] == 2) & (trip["purpose"] == "school"), "purpose"
    ] = "univ"

    # For people making school trips, they should be coded as some sort of student
    filter = (trip["purpose"] == "school") & (~trip["ptype"].isin([6, 7, 8]))
    person_list = trip.loc[filter, "person_id"]

    person.loc[
        (person["person_id"].isin(person_list)) & (person["age"] <= 5), "ptype"
    ] = 8
    person.loc[
        (person["person_id"].isin(person_list))
        & (person["age"] > 5)
        & (person["age"] < 16),
        "ptype",
    ] = 7
    person.loc[
        (person["person_id"].isin(person_list))
        & (person["age"] > 16)
        & (person["age"] < 21),
        "ptype",
    ] = 6

    # For people making work trips, they must be full- or part-time worker, or university or driving age student
    filter = (trip["purpose"] == "work") & (~trip["ptype"].isin([1, 2, 3, 6]))
    person_list = trip.loc[filter, "person_id"]
    person.loc[
        (person["person_id"].isin(person_list)) & (person["pemploy"] == 1), "ptype"
    ] = 1
    # No school age or preschool kid should be making a work trip
    # Remove these tours entirely
    filter = (tour["tour_type"] == "work") & (tour["ptype"].isin([7, 8]))
    logger.info(f"Dropped {len(tour[filter])} tours: under 16 making work tours")
    tour = tour[~filter]


    person.loc[
        (person["person_id"].isin(person_list)) & (person["pemploy"] == 2) & (person['age'] > 17), "ptype"
    ] = 2
    person.loc[
        (person["person_id"].isin(person_list))
        & (person["pemploy"] == 3)
        & (person["pstudent"] == 2),
        "ptype",
    ] = 3



    # For people making univ trips, they must be full-or part time worker or university student
    filter = (trip["purpose"] == "univ") & (~trip["ptype"].isin([1, 2, 3]))
    person_list = trip.loc[filter, "person_id"]
    person.loc[
        (person["person_id"].isin(person_list)) & (person["pemploy"] == 1), "ptype"
    ] = 1
    person.loc[
        (person["person_id"].isin(person_list)) & (person["pemploy"] == 2), "ptype"
    ] = 2
    person.loc[
        (person["person_id"].isin(person_list))
        & (person["pemploy"] == 3)
        & (person["pstudent"] == 2),
        "ptype",
    ] = 3

    # Drop trips without a valid purpose
    trip = trip[trip["purpose"] != "-1"]

    # Make sure a subtour's parent tour still exists. If not, remove
    tour["drop"] = 0
    tour.loc[
        (tour["tour_category"] == "atwork")
        & ~(tour["parent_tour_id"].isin(tour.tour_id)),
        "drop",
    ] = 1
    logger.info(
        f'Dropped {len(tour[tour["drop"] == 1])} tours: parent tour was removed/missing'
    )
    tour = tour[tour["drop"] == 0]

    # joint_tour_frequency.py requires that PNUM=1 is the primary joint tour maker
    # FIXME: This is a temporary assumption in activitysim that should be corrected
    # I don't know if it matters at this point except for household interaction models?

    # Merge person number to tours
    if "PNUM" not in tour.columns:
        tour = tour.merge(person[["person_id", "PNUM"]], on="person_id", how="left")
    # Find where PNUM=1 is not present in joint tour
    # This throws an error in joint_tour_frequency.py
    df = tour[tour["tour_category"] == "joint"].groupby("tour_id").min()[["PNUM"]]
    # For now, drop these joint tours completely (?)
    # FIXME: could possibly change the person number ordering or change activitysim restrictions
    _filter = tour["tour_id"].isin(df[df["PNUM"] > 1].index)
    logger.info(f"Dropped {len(tour[_filter])} joint tours: PNUM=1 not present in tour")
    tour = tour[~_filter]
    joint_tour_participants = joint_tour_participants[
        ~joint_tour_participants["tour_id"].isin(df[df["PNUM"] > 1].index)
    ]
    # tour = tour[tour_cols]

    # Joint tours can't be for work, school, or escort purposes
    # For now, remove these
    # FIXME: make them non-joint?
    filter = ((tour['tour_type'].isin(['work','school','escort']))&(tour['tour_category']=='joint'))
    logger.info(f"Dropped {len(tour[filter])} tours: joint tour cannot be work, school, or escort")
    tour = tour[~filter]

    # Set local person order in household to work with joint tour estimation
    # joint_tour_frequency.py assumes PNUM==1 should have a joint tour
    # Re-sort PNUM in person file based on joint_tour_participants file
    # Sort by most joint tours taken in household
    person_joint_tours = (
        joint_tour_participants.groupby("person_id")
        .count()["household_id"]
        .reset_index()
    )
    person_joint_tours.rename(
        columns={"household_id": "joint_tour_count"}, inplace=True
    )
    person = person.merge(person_joint_tours, how="left", on="person_id").fillna(0)
    person = person.sort_values(["household_id", "joint_tour_count"], ascending=False)
    for household_id in person.household_id.unique():
        person.loc[person["household_id"] == household_id, "PNUM"] = range(
            1, 1 + len(person[person["household_id"] == household_id])
        )

    # Update joint_tour_participants with new PNUM
    joint_tour_participants.drop("participant_num", axis=1, inplace=True)
    joint_tour_participants = joint_tour_participants.merge(
        person[["person_id", "PNUM"]], how="left"
    )

    # Tour records for joint tours only have info on primary trip makers; update these since we changed primary tour maker (PNUM==1)
    df = joint_tour_participants[joint_tour_participants["PNUM"] == 1]
    joint_tour_participants.rename(
        columns={"PNUM": "participant_num"}, inplace=True
    )  # reset to original col name
    # Make sure joint tour participants align with joint tours

    tour = tour.merge(df[["tour_id", "person_id"]], how="left", on="tour_id")
    tour["person_id_y"].fillna(tour["person_id_x"], inplace=True)
    tour.drop("person_id_x", axis=1, inplace=True)
    tour.rename(columns={"person_id_y": "person_id"}, inplace=True)

    tour = tour[~tour["tour_id"].duplicated()]

    tour.loc[(tour['tour_category'] == 'atwork') & ~(tour['parent_tour_id'].isin(tour.tour_id)), 'drop'] = 1
    logger.info(f"Dropped {len(tour[tour['drop'] == 1])} tours: atwork tour missing parent tour")
    tour = tour[tour['drop']==0]

    # Make sure trips, tours, and joint_tour_participants align
    trip = trip[trip["tour_id"].isin(tour["tour_id"])]
    tour = tour[tour["tour_id"].isin(trip["tour_id"])]
    joint_tour_participants = joint_tour_participants[
        joint_tour_participants["tour_id"].isin(tour["tour_id"])
    ]
    households = households[households["household_id"].isin(person["household_id"])]

    person = person[config["person_columns"]]
    tour = tour[config["tour_columns"]]
    trip = trip[config["trip_columns"]]
    households = households[config["hh_columns"]]

    households, person, tour, trip, joint_tour_participants = remove_hh_with_missing_work_location(households, person, tour, trip, joint_tour_participants, logger)
    person = recode_missing_school_location_to_home(households, person, tour, trip)


    joint_tour_participants.to_csv(
        os.path.join(config["output_dir"], "cleaned","survey_joint_tour_participants.csv"),
        index=False,
    )
    tour.to_csv(
        os.path.join(config["output_dir"], "cleaned", "survey_tours.csv"), index=False
    )
    households.to_csv(
        os.path.join(config["output_dir"], "cleaned", "survey_households.csv"), index=False
    )
    person.to_csv(
        os.path.join(config["output_dir"], "cleaned", "survey_persons.csv"), index=False
    )
    trip.to_csv(
        os.path.join(config["output_dir"], "cleaned", "survey_trips.csv"), index=False
    )

    # Conclude log
    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------RUN ENDING--------------------")
    logger.info("TOTAL RUN TIME %s" % str(elapsed_total))
