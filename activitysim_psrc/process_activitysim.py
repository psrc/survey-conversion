import os
import toml
from activitysim_psrc import locate_parcels, convert_format, clean, infer

main_config = toml.load("main_configuration.toml")
activitysim_config = toml.load(main_config["activitysim_configuration"])

if activitysim_config["run_locate_parcels"]:
    locate_parcels.locate_parcels(activitysim_config)

if activitysim_config["run_convert_format"]:
    convert_format.convert_format(activitysim_config)

if activitysim_config["run_clean"]:
    clean.clean(activitysim_config)

if activitysim_config["run_infer"]:
    infer.infer(
        activitysim_config["asim_config_dir"],
        activitysim_config["output_dir_final"],
        activitysim_config["output_dir_final"],
    )


print("done")
