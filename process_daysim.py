from daysim import locate_parcels, convert_format, clean, attach_skims, data_validation, unclone
import os
import toml

main_config = toml.load("main_configuration.toml")
daysim_config = toml.load(main_config["daysim_configuration"])

if daysim_config["run_locate_parcels"]:
    locate_parcels.locate_parcels(daysim_config)

if daysim_config["run_convert_format"]:
    convert_format.convert_format(daysim_config)

if daysim_config["run_clean"]:
    clean.clean(daysim_config)

if daysim_config["run_data_validation"]:
    data_validation.data_validation(daysim_config)

if daysim_config["run_attach_skims"]:
    attach_skims.attach_skims(daysim_config)

if daysim_config["run_unclone"]:
   unclone.unclone(daysim_config)

print("done")
