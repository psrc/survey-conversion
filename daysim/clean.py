import os
import pandas as pd
import toml

def clean(config):

    trip = pd.read_csv(os.path.join(config["output_dir"], "_trip.tsv"), sep="\t")
    tour = pd.read_csv(os.path.join(config["output_dir"], "_tour.tsv"), sep="\t")
    hh = pd.read_csv(os.path.join(config["output_dir"], "_household.tsv"), sep="\t")
    person = pd.read_csv(os.path.join(config["output_dir"], "_person.tsv"), sep="\t")
    person_day = pd.read_csv(os.path.join(config["output_dir"], "_person_day.tsv"), sep="\t")
    hh_day = pd.read_csv(os.path.join(config["output_dir"], "_household_day.tsv"), sep="\t")

    # Create new output directory for survey records with skims attached, if needed
    cleaned_output_dir = os.path.join(config["output_dir"], "cleaned")
    
    for my_path in [cleaned_output_dir]:
        if not os.path.exists(my_path):
            os.makedirs(my_path)

    # add some missing columns
    if 'puwmode' not in person.columns:
        person['puwmode'] = -1

    # Select only trips that match tours
    trip = trip[trip['tour'] >= 0]

    # Update trip mode to combined all access modes
    trip.loc[trip['mode'] == 7, 'mode'] = 6

    
    # Set all travel days to 1
    trip["day"] = 1
    tour["day"] = 1
    hh_day["day"] = 1
    hh_day["dow"] = 1
    person_day["day"] = 1

    # Update person weights to reflect duplicated records
    # Since person records were duplicated we need to divide the 
    # original weight by the number of days for which duplicated records are available
    # First, we trim person records to only include those on which valid
    # person day records are available on Monday-Thursday
    person = person[person['person_id'].isin(person_day['person_id'])]
    
    # Number of valid days found by occurence of original person ID duplicates 
    person = person.merge(
        person['person_id_original'].value_counts().reset_index(), 
        on='person_id_original', 
        how='left'
    )
    person['psexpfac'] = person['psexpfac']/person['count']

    # Re-calculate household weights in the same way
    hh = hh[hh['hhno'].isin(person_day['hhno'])]
    hh = hh.merge(
        hh['hhid_elmer'].value_counts().reset_index(), 
        on='hhid_elmer', 
        how='left'
    )
    hh['hhexpfac'] = hh['hhexpfac']/hh['count']


    # Re-calculate tour weights
    # Tour weights are to be taken as the average of trip weights
    # FIXME: move this into the standard tour creation module
    df = trip[['tour','trexpfac']].groupby('tour').mean().reset_index()
    df.rename(columns={'trexpfac': 'toexpfac'}, inplace=True)
    tour.drop('toexpfac', axis=1, inplace=True)
    tour = tour.merge(df, on='tour', how='left')

    # Update person day weights
    person_day.drop('pdexpfac', axis=1, inplace=True)
    person_day = person_day.merge(
        person[['person_id','psexpfac']], 
        on='person_id', 
        how='left'
    )
    person_day.rename(columns={'psexpfac': 'pdexpfac'}, inplace=True)

    # Update household day weights
    hh_day.drop('hdexpfac', axis=1, inplace=True)
    hh_day = hh_day.merge(
        hh[['hhno','hhexpfac']], 
        on='hhno', 
        how='left'
    )
    hh_day.rename(columns={'hhexpfac': 'hdexpfac'}, inplace=True)

    # Do some further flagging
    # Flag trips that have missing O/Ds, purposes, etc.



    # Select tours without errors/issues

    # write these out as the final versions and have skims attached
    trip.to_csv(os.path.join(cleaned_output_dir, "_trip.tsv"), sep="\t", index=False)
    tour.to_csv(os.path.join(cleaned_output_dir, "_tour.tsv"),sep="\t", index=False)
    hh.to_csv(os.path.join(cleaned_output_dir, "_household.tsv"),sep="\t", index=False)
    person.to_csv(os.path.join(cleaned_output_dir, "_person.tsv"),sep="\t", index=False)
    person_day.to_csv(os.path.join(cleaned_output_dir, "_person_day.tsv"),sep="\t", index=False)
    hh_day.to_csv(os.path.join(cleaned_output_dir, "_household_day.tsv"),sep="\t", index=False)