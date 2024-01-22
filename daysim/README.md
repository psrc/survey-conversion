# Daysim Conversion 
Run **process_survey.py** to run all scripts. **configuration.toml** manages which scripts are run and contains all input assumptions.

## Scripts
- **locate_parcels.py** snaps survey records such as trip origin/destination, home, school, and work location to appropriate parcels nearest reported coordinates 
- **convert_format.py** converts the geolocated survey data to Daysim format, builds tour and person day files
- **data_validation.py** checks the output of survey conversion to ensure results meet expected criteria and coerces data into proper formats
- **summary/write_summary_nb.py** generates an HTML comparison of two Daysim-formatted survey files. These files are built based off the notebooks in the summary directory.
    - set file locations, names, etc. in summary/summary_configuration.toml   
