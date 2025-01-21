import os, datetime
import pandas as pd
import toml
from modules import util
from daysim import logcontroller

def clean(config):

    logger = logcontroller.setup_custom_logger("clean_logger.txt", config)
    logger.info("--------------------clean.py STARTING--------------------")
    start_time = datetime.datetime.now()

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
    # person = person.merge(
    #     person['person_id_original'].value_counts().reset_index(), 
    #     on='person_id_original', 
    #     how='left'
    # )
    df = person['person_id_original'].value_counts().reset_index()
    df.rename(columns={'person_id_original': 'count', 'index': 'person_id_original'}, inplace=True)
    # person['psexpfac'] = person['psexpfac']/person['count']
    person = person.merge(
        df, 
        on='person_id_original', 
        how='left'
    )
    person['psexpfac_original'] = person['psexpfac'].copy()
    person['psexpfac'] = person['psexpfac']/person['count']

    # Re-calculate household weights in the same way
    hh = hh[hh['hhno'].isin(person_day['hhno'])]
    df = hh['hhid_elmer'].value_counts().reset_index()
    df.rename(columns={'hhid_elmer': 'count', 'index': 'hhid_elmer'}, inplace=True)
    hh = hh.merge(
        df, 
        on='hhid_elmer', 
        how='left'
    )
    hh['hhexpfac_original'] = hh['hhexpfac'].copy()
    hh['hhexpfac'] = hh['hhexpfac']/hh['count']

    # Update person day weights
    person_day.rename(columns={'pdexpfac':'pdexpfac_original'}, inplace=True)
    person_day = person_day.merge(
        person[['person_id','psexpfac']], 
        on='person_id', 
        how='left'
    )
    person_day.rename(columns={'psexpfac': 'pdexpfac'}, inplace=True)

    # Update household day weights
    hh_day.rename(columns={'hdexpfac': 'hdexpfac_original'},inplace=True)
    hh_day = hh_day.merge(
        hh[['hhno','hhexpfac']], 
        on='hhno', 
        how='left'
    )
    hh_day.rename(columns={'hhexpfac': 'hdexpfac'}, inplace=True)

    # Do some further flagging
    # All workers should have a valid work TAZ/parcel
    # Try to use their most visited work location as their usual workplace
    # This is important for location choice models
    # Use tour file to get work departure/arrival time and mode
    work_tours = tour[tour["pdpurp"] == 1]
    no_pwtaz_df = person.loc[(person['pwtyp'].isin([1,2])) & (person['worker_type'].isin(['commuter','telecommuter'])) & (person['pwtaz']==-1)]

    df = work_tours[work_tours['person_id'].isin(no_pwtaz_df['person_id'])]

    # Calculate time spent at destination and choose the one where they spent the longest time
    df['time_at_dest'] = df['tlvdest']-df['tardest']
    df = df.sort_values('time_at_dest', ascending=False).drop_duplicates(['person_id'])
    person = person.merge(df[['person_id','tdtaz','tdpcl']], on='person_id', how='left')

    # Replace usual work location with these values on the person file
    person.loc[person['person_id'].isin(no_pwtaz_df.person_id), 'pwtaz'] = person['tdtaz'].fillna(-1)
    person.loc[person['person_id'].isin(no_pwtaz_df.person_id), 'pwpcl'] = person['tdpcl'].fillna(-1)

    # If no workplace TAZ and no work tours, change the person type to non-worker
    person.loc[person['person_id'].isin(no_pwtaz_df.person_id) & (person['pwpcl']==-1), 'pwtyp'] = 0
    # Change person type to non-worker based on age
    # Non-working adult >65
    person.loc[person['person_id'].isin(no_pwtaz_df.person_id) & (person['pwpcl']==-1) & (person['pagey']>65), 'pptyp'] = 3
    # Non-working adult <= 65 (and not a student)
    person.loc[person['person_id'].isin(no_pwtaz_df.person_id) & (person['pwpcl']==-1) & 
               (person['pagey']<=65) & (person['pstyp']==0), 'pptyp'] = 4
    # University student
    person.loc[person['person_id'].isin(no_pwtaz_df.person_id) & (person['pwpcl']==-1) & 
               (person['pagey']>=18) & (person['pstyp']!=0), 'pagey'] = 5
    # High school student if under 18
    person.loc[person['person_id'].isin(no_pwtaz_df.person_id) & (person['pwpcl']==-1) & (person['pagey']<18), 'pptyp'] = 6

    if config["scale_totals"]:
        # Scale weights so the smaller, cleaned dataset matches totals
        trip_original = util.load_elmer_table(config["elmer_trip_table"], 
                                                sql="SELECT trip_weight FROM "+config["elmer_trip_table"]+\
                                                    " WHERE survey_year in "+str(config['survey_year']))
        hh_original = util.load_elmer_table(config["elmer_hh_table"], 
                                                sql="SELECT hh_weight FROM "+config["elmer_hh_table"]+\
                                                    " WHERE survey_year in "+str(config['survey_year']))
        person_original = util.load_elmer_table(config["elmer_person_table"], 
                                                sql="SELECT person_weight FROM "+config["elmer_person_table"]+\
                                                    " WHERE survey_year in "+str(config['survey_year']))
        
        trip_wt_scale = trip_original['trip_weight'].sum()/trip['trexpfac'].sum()
        hh_wt_scale = hh_original['hh_weight'].sum()/hh['hhexpfac'].sum()
        person_wt_scale = person_original['person_weight'].sum()/person['psexpfac'].sum()

        trip['trexpfac'] = trip['trexpfac']*(trip_wt_scale)
        hh['hhexpfac'] = hh['hhexpfac']*(hh_wt_scale)
        hh_day['hdexpfac'] = hh_day['hdexpfac']*(hh_wt_scale)
        person['psexpfac'] = person['psexpfac']*(person_wt_scale)
        person_day['pdexpfac'] = person_day['pdexpfac']*(person_wt_scale)

        logger.info(f"Trip weights scaled by: {trip_wt_scale} to match Elmer totals")
        logger.info(f"Household weights scaled by: {hh_wt_scale} to match Elmer totals")
        logger.info(f"Person weights scaled by: {person_wt_scale} to match Elmer totals")
    
    # Re-calculate tour weights
    # Tour weights are to be taken as the average of trip weights
    df = trip[['tour','trexpfac']].groupby('tour').mean().reset_index()
    df.rename(columns={'trexpfac': 'toexpfac'}, inplace=True)
    tour.drop('toexpfac', axis=1, inplace=True)
    tour = tour.merge(df, on='tour', how='left')

    # write these out as the final versions and have skims attached
    trip.to_csv(os.path.join(cleaned_output_dir, "_trip.tsv"), sep="\t", index=False)
    tour.to_csv(os.path.join(cleaned_output_dir, "_tour.tsv"),sep="\t", index=False)
    hh.to_csv(os.path.join(cleaned_output_dir, "_household.tsv"),sep="\t", index=False)
    person.to_csv(os.path.join(cleaned_output_dir, "_person.tsv"),sep="\t", index=False)
    person_day.to_csv(os.path.join(cleaned_output_dir, "_person_day.tsv"),sep="\t", index=False)
    hh_day.to_csv(os.path.join(cleaned_output_dir, "_household_day.tsv"),sep="\t", index=False)

    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------convert_format.py ENDING--------------------")
    logger.info("convert_format.py RUN TIME %s" % str(elapsed_total))