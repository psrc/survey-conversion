from urbansim import locate_parcels_urbansim
from urbansim import convert_survey_urbansim
import os
import toml

main_config = toml.load("main_configuration.toml")
urbansim_config = toml.load(main_config["urbansim_configuration"])

if urbansim_config["run_locate_parcels"]:
    locate_parcels_urbansim.locate_parcels_urbansim(urbansim_config)

if urbansim_config["run_convert_format"]:
    convert_survey_urbansim.convert_format_urbansim(urbansim_config)

print("done")
