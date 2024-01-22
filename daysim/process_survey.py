import locate_parcels
import convert_format
import attach_skims
import data_validation
import configuration
import os
import toml

config = toml.load("configuration.toml")

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
