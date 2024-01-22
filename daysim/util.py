# Create zone number dictionary from project
# NOTE: use the Soundcast environment to run this (model_py3)
import json
from EmmeProject import *
from lookup import run_root

project_dir = os.path.join(run_root, r"projects\8to9\8to9.emp")
my_project = EmmeProject(run_root)

zones = my_project.current_scenario.zone_numbers
dictZoneLookup = dict((value, index) for index, value in enumerate(zones))

# Write results to working directory
json.dump(dictZoneLookup, open("zone_dict.txt", "w"))
