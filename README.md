# Survey Conversion 
This repository contains code that converts Household Travel Survey (HTS) data to formats used by the Soundcast travel model. Data can be converted into formats used by Daysim (current model core of Soundcast) and Activitysim (under development). The Daysim modules are fully up to date, but Activitysim needs additional work to fit with the paradigm. The working version of the Activitysim conversion should be accessed from the [psrc_activitysim](https://github.com/psrc/psrc_activitysim/tree/main/scripts/survey_conversion) repo. 

# Code
Script specific to **activitysim** and daysim are in their respective folders. The **modules** folders contains functions shared across scripts. Currently Activitysim and Daysim are run and managed separately so they are described independently. 

## Daysim
The file **daysim_configuration.toml** contains all settings and input assumptions to run Daysim conversions. This includes which modules to run, output directories, and data dictionaries specific to Daysim and the PSRC HTS. 

The script **process_daysim.py** run any of the following scripts from the daysim directory, depending on configuration settings:

- **locate_parcels.py** snaps survey records such as trip origin/destination, home, school, and work location to appropriate parcels nearest reported coordinates. 
- **convert_format.py** converts the geolocated survey data to Daysim format, builds tour and person day files
- **data_validation.py** checks the output of survey conversion to ensure results meet expected criteria and coerces data into proper formats
- **attach_skims.py** add model skim values to trips, tours, and usual work and school measures. 

A set of summaries is available to compare different versions of formatted survey outputs. 
- **summary/write_summary_nb.py** generates an HTML comparison of two Daysim-formatted survey files. These files are built based off the notebooks in the summary directory.
    - set file locations, names, etc. in summary/summary_configuration.toml   

## Activitysim
The set of Activitysim is under development to synchronize with this codebase. The current working version is run directly from a script in the [psrc_activitysim](https://github.com/psrc/psrc_activitysim/blob/main/scripts/survey_conversion/activitysim_survey_conversion_multiday.py) repo.
