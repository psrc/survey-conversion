import os
import toml
from activitysim_psrc import locate_parcels, convert_format, clean, data_validation, infer, unclone
from activitysim.core import workflow
from activitysim.abm.models.util import canonical_ids as cid
from activitysim import cli as client
import sys
import argparse

main_config = toml.load("main_configuration.toml")
activitysim_config = toml.load(main_config["activitysim_configuration"])

sys.argv.append('--working_dir')
sys.argv.append(activitysim_config['asim_state_wd'])
parser = argparse.ArgumentParser()
client.run.add_run_args(parser)
args = parser.parse_args()
state = workflow.State()
state.logging.config_logger(basic=True)
state = client.run.handle_standard_args(state, args)  # possibly update injectables

if activitysim_config["run_locate_parcels"]:
    locate_parcels.locate_parcels(activitysim_config)

if activitysim_config["run_convert_format"]:
    convert_format.convert_format(activitysim_config, state)

if activitysim_config["run_clean"]:
    clean.clean(activitysim_config, state)

if activitysim_config["run_data_validation"]:
    data_validation.data_validation(activitysim_config)

if activitysim_config["run_infer"]:
    infer.infer(
        activitysim_config["asim_config_dir"],
        os.path.join(activitysim_config["output_dir"], "cleaned"),
        os.path.join(activitysim_config["output_dir"], "cleaned"),
        state
    )

if activitysim_config["run_unclone"]:
   unclone.unclone(activitysim_config)




print("done")
