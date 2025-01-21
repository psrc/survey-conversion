# Survey Conversion 
This repository contains code that converts Household Travel Survey (HTS) data to formats used by the Soundcast travel model. Data can be converted into formats used by Daysim (current model core of Soundcast) and Activitysim (under development). T
Scripts specific to **activitysim** and **daysim** are in their respective folders. The **modules** folders contains functions shared across scripts. Currently Activitysim and Daysim are run and managed separately so they are described independently. In general scripts for both model versions work similarly. 

## Daysim
The file **daysim_configuration.toml** contains all settings and input assumptions to run Daysim conversions. This includes which modules to run, output directories, and data dictionaries specific to Daysim and the PSRC HTS. 

### Setup
A virtual environment is included at **daysim/environment.yml**. Install and activate before running the scripts. 

### Running
The script **process_daysim.py** runs the following scripts from the daysim directory, depending on configuration settings:

- **locate_parcels.py** snaps survey records such as trip origin/destination, home, school, and work location to appropriate parcels nearest reported coordinates. 
- **convert_format.py** converts the geolocated survey data to Daysim format, builds tour and person day files
- **clean.py** filters out trip and tour data with issues flagged during conversion or in this cleaning process. 
- **data_validation.py** checks the output of survey conversion to ensure results meet expected criteria and coerces data into proper formats
- **attach_skims.py** add model skim values to trips, tours, and usual work and school measures.

The code is designed to be used with and without PSRC databases. The scripts locate_parcels.py and attach_skims.py will require additional inputs, but convert_format.py should be able to rely solely on the geolocated survey records and inputs found in this directory. Most conversion parameters are specified in daysim_configuration.toml and in expression files in daysim/inputs. These confirgurable parameters were designed to update conversions without needing to access the code directly. While there are some PSRC-specific assumptions in the scripts such as convert_format.py, most changes should only need to be made to this input files rather than the code itself. 

### Summaries
A set of summaries is available to compare different versions of formatted survey outputs. 
- **summary/write_summary_nb.py** generates an HTML comparison of two Daysim-formatted survey files. These files are built based off the notebooks in the summary directory.
    - set file locations, names, etc. in summary/summary_configuration.toml   

## Activitysim
Activitysim is designed to run similarly to Daysim as described above. The file **activitysim_configuration.toml** contains all settings and input assumptions to run Activitysim conversions. This includes which modules to run, output directories, and data dictionaries specific to Daysim and the PSRC HTS. 

### Setup
Activitysim must be installed and an activitysim virtual environment activated to run this script. The script uses built-in modules from Activitysim to filter and clean data. The activitysim script should contain most libraries required to run the script except for parcel location, which can be turned off in the activitysim_configuration file. A virtual enviornment install file will be provided with this repository soon. 

### Running
The script **process_activitysim.py** runs the following scripts from the activitysim directory, depending on configuration settings:

- **locate_parcels.py** snaps survey records such as trip origin/destination, home, school, and work location to appropriate parcels nearest reported coordinates. 
- **convert_format.py** converts the geolocated survey data to Daysim format, builds tour and person day files
- **clean.py** filters out trip and tour data with issues flagged during conversion or in this cleaning process. 
- **infer.py** is an Activitysim script copied into this repository that creates the final formatted inputs used to build estimation data bundles.

### Outputs
Outputs will be stored in the output directory set in activitysim_configuration.toml. The first set of files produced by geolocation and inital format conversion will be available at this location. These files should contain as many records as possible and can be used for validation purposes. Trips and tours have error flags that will be used to filter in the following steps, so that files are acceptable for estimation. When clean.py is run, a folded title **cleaned** will be available in the output directory. This directory contains the filtered and cleaned files from both the output of clean.py and from the final results of infer.py

The code is designed to be used with and without PSRC databases. The scripts locate_parcels.py and attach_skims.py will require additional inputs, but convert_format.py should be able to rely solely on the geolocated survey records and inputs found in this directory. Most conversion parameters are specified in daysim_configuration.toml and in expression files in daysim/inputs. These confirgurable parameters were designed to update conversions without needing to access the code directly. While there are some PSRC-specific assumptions in the scripts such as convert_format.py, most changes should only need to be made to this input files rather than the code itself. 
