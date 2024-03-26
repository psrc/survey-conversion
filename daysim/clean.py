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

    # Make sure person day, trip, and tours match this slimmed down set of persons

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