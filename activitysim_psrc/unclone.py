import os
import pandas as pd
import toml

def unclone(config):
    for outdir in config['unclone_path_list']:
        uncloned_output_dir = os.path.join(outdir, "uncloned")
        for my_path in [uncloned_output_dir]:
            if not os.path.exists(my_path):
                os.makedirs(my_path)

        trip = pd.read_csv(os.path.join(outdir,"survey_trips.csv"))
        tour = pd.read_csv(os.path.join(outdir,"survey_tours.csv"))
        hh = pd.read_csv(os.path.join(outdir, "survey_households.csv"))
        person = pd.read_csv(os.path.join(outdir, "survey_persons.csv"))
        jt_participants = pd.read_csv(os.path.join(outdir, "survey_joint_tour_participants.csv"))

        # Remove duplicate households and person for household- and person-level model estimation 
        #hh_uncloned = hh[~hh[hh.columns.drop('household_id')].duplicated()]
        hh_uncloned = hh.groupby('household_id_original', as_index=False).first()

        # Reset weights to use the original values that haven't been factored for cloned persons/days
        hh_uncloned.rename(columns={'hh_weight': 'hh_weight_drop', 'hh_weight_original': 'hh_weight'}, inplace=True)

        for name_type in ['override','survey']:
            hh_uncloned.to_csv(os.path.join(outdir, "uncloned", name_type+"_households.csv"), index=False)

            person_uncloned = person[person['household_id'].isin(hh_uncloned['household_id'])]
            person_uncloned.rename(columns={'person_weight': 'person_weight_drop', 'person_weight_original': 'person_weight'}, inplace=True)
            person_uncloned.drop('person_weight_drop', inplace=True, axis=1)
            person_uncloned.to_csv(os.path.join(outdir, "uncloned", name_type+"_persons.csv"), index=False)

            trip_uncloned = trip[trip['household_id'].isin(hh_uncloned['household_id'])]
            trip_uncloned.rename(columns={'trip_weight': 'trip_weight_drop', 'trip_weight_original': 'trip_weight'}, inplace=True)
            trip_uncloned.drop('trip_weight_drop', inplace=True, axis=1)
            trip_uncloned.to_csv(os.path.join(outdir, "uncloned", name_type+"_trips.csv"), index=False)

            tour_uncloned = tour[tour['household_id'].isin(hh_uncloned['household_id'])]    
            tour_uncloned.to_csv(os.path.join(outdir, "uncloned", name_type+"_tours.csv"), index=False)

            jt_participants_uncloned = jt_participants[jt_participants['household_id'].isin(hh_uncloned['household_id'])]
            jt_participants_uncloned.to_csv(os.path.join(outdir, "uncloned", name_type+"_joint_tour_participants.csv"), index=False)