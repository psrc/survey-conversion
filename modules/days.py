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


def create(tour, person, trip, config):
    pday = pd.DataFrame()

    for person_rec in person["unique_person_id"].unique():
        # get this person's tours
        _tour = tour[tour["unique_person_id"] == person_rec]

        if len(_tour) > 0:
            # Loop through each day
            for day in _tour["day"].unique():
                day_tour = _tour[_tour["day"] == day]

                # prec_id = str(person_rec) + str(day)
                pday.loc[person_rec, "hhno"] = day_tour["hhno"].iloc[0]
                pday.loc[person_rec, "pno"] = day_tour["pno"].iloc[0]
                pday.loc[person_rec, "person_id"] = day_tour["person_id"].iloc[0]
                pday.loc[person_rec, "hhid_elmer"] = day_tour[
                    "hhid_elmer"
                ].iloc[0]
                pday.loc[person_rec, "unique_person_id"] = person_rec
                pday.loc[person_rec, "day"] = day
                pday.loc[person_rec, "pdexpfac"] = person[
                    person["unique_person_id"] == person_rec
                ].iloc[0]["psexpfac"]

                # Begin/End at home-
                # need to get from first and last trips of tour days
                pday.loc[person_rec, "beghom"] = 0
                pday.loc[person_rec, "endhom"] = 0
                _trip = trip[
                    (trip["unique_person_id"] == person_rec) & (trip["day"] == day)
                ]
                if _trip.iloc[0]["opurp"] == 0:
                    pday.loc[person_rec, "beghom"] = 1
                if _trip.iloc[-1]["dpurp"] == 0:
                    pday.loc[person_rec, "endhom"] = 1

                # Number of tours by purpose
                for purp_name, purp_val in config["purp_dict"].items():
                    pday.loc[person_rec, purp_name + "tours"] = len(
                        day_tour[day_tour["pdpurp"] == int(purp_val)]
                    )

                    # Number of stops
                    day_tour_purp = day_tour[day_tour["pdpurp"] == int(purp_val)]
                    if len(day_tour_purp) > 0:
                        nstops = day_tour_purp[["tripsh1", "tripsh2"]].sum().sum() - 2
                    else:
                        nstops = 0
                    pday.loc[person_rec, purp_name + "stops"] = nstops

                # Home based tours
                pday.loc[person_rec, "hbtours"] = len(day_tour)

                # Work-based tours (subtours)
                pday.loc[person_rec, "wbtours"] = day_tour["subtrs"].sum()

                # Work trips to usual workplace
                pday.loc[person_rec, "uwtours"] = len(
                    day_tour[day_tour["tdpcl"] == day_tour["pwpcl"]]
                )

    return pday
