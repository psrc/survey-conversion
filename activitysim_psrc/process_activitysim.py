import os
import toml
from activitysim_psrc import locate_parcels, convert_format, clean, data_validation, infer, unclone

main_config = toml.load("main_configuration.toml")
activitysim_config = toml.load(main_config["activitysim_configuration"])

if activitysim_config["run_locate_parcels"]:
    locate_parcels.locate_parcels(activitysim_config)

if activitysim_config["run_convert_format"]:
    convert_format.convert_format(activitysim_config)

if activitysim_config["run_clean"]:
    clean.clean(activitysim_config)

if activitysim_config["run_data_validation"]:
    data_validation.data_validation(activitysim_config)

if activitysim_config["run_infer"]:
    infer.infer(
        activitysim_config["asim_config_dir"],
        os.path.join(activitysim_config["output_dir"], "cleaned"),
        os.path.join(activitysim_config["output_dir"], "cleaned"),
    )

if activitysim_config["run_unclone"]:
   unclone.unclone(activitysim_config)




print("done")
