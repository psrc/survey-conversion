from daysim import locate_parcels
from daysim import convert_format
from daysim import attach_skims
from daysim import data_validation
import os
import toml

main_config = toml.load("main_configuration.toml")
daysim_config = toml.load(main_config["daysim_configuration"])

if daysim_config["run_locate_parcels"]:
    locate_parcels.locate_parcels(daysim_config)

if daysim_config["run_convert_format"]:
    convert_format.convert_format(daysim_config)

if daysim_config["run_data_validation"]:
    data_validation.data_validation(daysim_config)

if daysim_config["run_attach_skims"]:
    attach_skims.attach_skims(daysim_config)

if daysim_config["run_data_validation"]:
    data_validation.data_validation(daysim_config)

print("done")
