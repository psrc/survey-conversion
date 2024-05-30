import os
import pandas as pd
import toml

def unclone(config):
    uncloned_output_dir = os.path.join(config["output_dir"], 'cleaned', "uncloned")
    for my_path in [uncloned_output_dir]:
        if not os.path.exists(my_path):
            os.makedirs(my_path)

    trip = pd.read_csv(os.path.join(config["output_dir"], 'cleaned',"override_trips.csv"))
    tour = pd.read_csv(os.path.join(config["output_dir"], 'cleaned',"override_tours.csv"))
    hh = pd.read_csv(os.path.join(config["output_dir"],'cleaned', "override_households.csv"))
    person = pd.read_csv(os.path.join(config["output_dir"],'cleaned', "override_persons.csv"))
    jt_participants = pd.read_csv(os.path.join(config["output_dir"],'cleaned', "override_joint_tour_participants.csv"))

    # Remove duplicate households and person for household- and person-level model estimation 
    #hh_uncloned = hh[~hh[hh.columns.drop('household_id')].duplicated()]
    hh_uncloned = hh.groupby('household_id_original', as_index=False).first()
    hh_uncloned.to_csv(os.path.join(uncloned_output_dir, "override_households.csv"), index=False)
    hh_uncloned.to_csv(os.path.join(uncloned_output_dir, "survey_households.csv"), index=False)

    person_uncloned = person[person['household_id'].isin(hh_uncloned['household_id'])]
    person_uncloned.to_csv(os.path.join(uncloned_output_dir, "override_persons.csv"), index=False)
    person_uncloned.to_csv(os.path.join(uncloned_output_dir, "survey_persons.csv"), index=False)

    trip_uncloned = trip[trip['household_id'].isin(hh_uncloned['household_id'])]
    trip_uncloned.to_csv(os.path.join(uncloned_output_dir, "override_trips.csv"), index=False)
    trip_uncloned.to_csv(os.path.join(uncloned_output_dir, "survey_trips.csv"), index=False)

    tour_uncloned = tour[tour['household_id'].isin(hh_uncloned['household_id'])]    
    tour_uncloned.to_csv(os.path.join(uncloned_output_dir, "override_tours.csv"), index=False)
    tour_uncloned.to_csv(os.path.join(uncloned_output_dir, "survey_tours.csv"), index=False)

    jt_participants_uncloned = jt_participants[jt_participants['household_id'].isin(hh_uncloned['household_id'])]
    jt_participants_uncloned.to_csv(os.path.join(uncloned_output_dir, "override_joint_tour_participants.csv"), index=False)
    jt_participants_uncloned.to_csv(os.path.join(uncloned_output_dir, "survey_joint_tour_participants.csv"), index=False)