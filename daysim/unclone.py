import os
import pandas as pd
import toml

def unclone(config):
    uncloned_output_dir = os.path.join(config["output_dir"], 'cleaned', 'skims_attached', "uncloned")
    for my_path in [uncloned_output_dir]:
        if not os.path.exists(my_path):
            os.makedirs(my_path)

    trip = pd.read_csv(os.path.join(config["output_dir"], 'cleaned', 'skims_attached', "_trip.tsv"), sep="\t")
    tour = pd.read_csv(os.path.join(config["output_dir"], 'cleaned', 'skims_attached',"_tour.tsv"), sep="\t")
    hh = pd.read_csv(os.path.join(config["output_dir"],'cleaned', 'skims_attached', "_household.tsv"), sep="\t")
    person = pd.read_csv(os.path.join(config["output_dir"],'cleaned', 'skims_attached', "_person.tsv"), sep="\t")
    person_day = pd.read_csv(os.path.join(config["output_dir"],'cleaned', 'skims_attached', "_person_day.tsv"), sep="\t")
    hh_day = pd.read_csv(os.path.join(config["output_dir"],'cleaned', 'skims_attached', "_household_day.tsv"), sep="\t")

    # Remove duplicate households and person for household- and person-level model estimation 
    hh_uncloned = hh[~hh[hh.columns.drop('hhno')].duplicated()]
    hh_uncloned.rename(columns={'hhexpfac': 'hhexpfac_drop', 'hhexpfac_original': 'hhexpfac'}, inplace=True)
    hh_uncloned.drop('hhexpfac_drop', axis=1, inplace=True)
    hh_uncloned.to_csv(os.path.join(uncloned_output_dir, "_household.tsv"), sep="\t", index=False)

    person_uncloned = person[person['hhno'].isin(hh_uncloned['hhno'])]
    person_uncloned.rename(columns={'psexpfac': 'psexpfac_drop', 'psexpfac_original': 'psexpfac'}, inplace=True)
    person_uncloned.drop('psexpfac_drop', axis=1, inplace=True)
    person_uncloned.to_csv(os.path.join(uncloned_output_dir, "_person.tsv"), sep="\t", index=False)

    trip_uncloned = trip[trip['hhno'].isin(hh_uncloned['hhno'])]
    trip_uncloned.to_csv(os.path.join(uncloned_output_dir, "_trip.tsv"), sep="\t", index=False)

    tour_uncloned = tour[tour['hhno'].isin(hh_uncloned['hhno'])]    
    tour_uncloned.to_csv(os.path.join(uncloned_output_dir, "_tour.tsv"), sep="\t", index=False)

    person_day_uncloned = person_day[person_day['hhno'].isin(hh_uncloned['hhno'])]
    person_day_uncloned.rename(columns={'pdexpfac': 'pdexpfac_drop', 'pdexpfac_original': 'pdexpfac'}, inplace=True)
    person_day_uncloned.drop('pdexpfac_drop', axis=1, inplace=True)
    person_day_uncloned.to_csv(os.path.join(uncloned_output_dir, "_person_day.tsv"), sep="\t", index=False)

    household_day_uncloned = hh_day[hh_day['hhno'].isin(hh_uncloned['hhno'])]
    household_day_uncloned.rename(columns={'hdexpfac': 'hdexpfac_drop', 'hdexpfac_original': 'hdexpfac'}, inplace=True)
    household_day_uncloned.drop('hdexpfac_drop', axis=1, inplace=True)
    household_day_uncloned.to_csv(os.path.join(uncloned_output_dir, "_household_day.tsv"), sep="\t", index=False)