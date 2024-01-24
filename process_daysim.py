from daysim import locate_parcels
from daysim import convert_format
from daysim import attach_skims
from daysim import data_validation
import configuration
import os
import toml

config = toml.load("daysim_configuration.toml")

if config["run_locate_parcels"]:
    locate_parcels.locate_parcels()

if config["run_convert_format"]:
    convert_format.convert_format()

if config["run_data_validation"]:
    data_validation.data_validation()

if config["run_attach_skims"]:
    attach_skims.attach_skims()

if config["run_data_validation"]:
    data_validation.data_validation()

print("done")
