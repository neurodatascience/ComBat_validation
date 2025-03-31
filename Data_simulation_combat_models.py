import os
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/d-ComBat_project")
from ppmi_age_sex_case_aseg.bootstrap_helper import neuro_combat_train,d_combat_train
import json

default_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to simulation config JSON file in Combat Sites")
args = parser.parse_args()

parameter_path = args.config
with open(parameter_path, "r") as f:
    config = json.load(f)


# x=24
# parameter_path=os.path.join(default_path,f"min_points{x}",
#                             f"Homogeneous_Homogeneous_Homogeneous_nonlinear_N{x*5}_G3_I5_Gamma4",
#                             "simulation.json")

# with open(parameter_path, "r") as f:
#     config = json.load(f)

# Access values
sampling_type = config["sampling_type"]
sex_type=config["sex_type"]
age_type = config["age_type"]
effect_type = config["effect_type"]
N = config["N"]
G = config["G"]
I = config["I"]
gamma_scale = config["gamma_scale"]
smallest_sample_size=config["smallest_sample_size"]

file_name=f'{sampling_type}_{sex_type}_{age_type}_{effect_type}_N{N}_G{G}_I{I}_Gamma{gamma_scale}'
results_common_path=os.path.join(default_path,
                                f"min_points{smallest_sample_size}",
                                f'{file_name}')

data=pd.read_csv(os.path.join(results_common_path,f"{file_name}.csv"))
print(data.columns)

removed_columns = [col for col in data.columns if "ground_truth" in col or "epsilon" in col]

data=data.drop(columns=removed_columns)

output_n=neuro_combat_train(data)

output_d=d_combat_train(data,results_common_path)

with open(os.path.join(results_common_path,"output_n.pkl"),'wb') as f:
    pickle.dump(output_n,f)
with open(os.path.join(results_common_path,"output_d.pkl"),'wb') as f:
    pickle.dump(output_d,f)
