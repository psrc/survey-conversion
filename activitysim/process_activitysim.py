from activitysim import locate_parcels
from activitysim import convert_format
# from daysim import attach_skims
# from daysim import data_validation
import os
import toml

main_config = toml.load("main_configuration.toml")
activitysim_config = toml.load(main_config["activitysim_coniguration"])

if activitysim_config["run_locate_parcels"]:
    locate_parcels.locate_parcels(activitysim_config)

if activitysim_config["run_convert_format"]:
    convert_format.convert_format(activitysim_config)

# if daysim_config["run_data_validation"]:
#     data_validation.data_validation(daysim_config)

# if daysim_config["run_attach_skims"]:
#     attach_skims.attach_skims(daysim_config)

# if daysim_config["run_data_validation"]:
#     data_validation.data_validation(daysim_config)

print("done")
