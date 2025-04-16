import os
import pandas as pd
import pickle
import sys
from helper import neuro_combat_train,d_combat_train

#specify your own working directory
default_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--store_folder", type=str)
parser.add_argument("--sampling_type", type=str)
parser.add_argument("--simulation_times", type=int)
parser.add_argument("--sex_type", type=str)
parser.add_argument("--age_type", type=str)
parser.add_argument("--effect_type", type=str)
parser.add_argument("--G", type=int)
parser.add_argument("--I", type=int)
parser.add_argument("--gamma_scale", type=float)
parser.add_argument("--N", type=int)
parser.add_argument("--smallest_sample_size_of_batches", type=int)

args = parser.parse_args()
config = vars(args)
###*************************#######################################################
# Access values
store_folder=config["store_folder"]
sampling_type = config["sampling_type"]
sex_type=config["sex_type"]
age_type = config["age_type"]
effect_type = config["effect_type"]
N = config["N"]
G = config["G"]
I = config["I"]
gamma_scale = config["gamma_scale"]
min_points=config["smallest_sample_size_of_batches"]
simulation_times=config["simulation_times"]

file_name=f'N{N}'
results_common_path=os.path.join(default_path,
                                store_folder,
                                f'{file_name}')

# There are two scenarios:
# 1. Multiple simulations exist, each stored in its own subfolder named 'simulationX'.
# 2. A single simulation is present directly in the main results folder.
file_names1=os.listdir(results_common_path)
file_names1=sorted(file_names1, key= lambda name:int(name.split("_")[1]))
if any("simulation" in name for name in file_names1):
    for f in file_names1: 
        try:
            path1=os.path.join(results_common_path,f)
            data = pd.read_csv(os.path.join(path1, "data.csv"))
            removed_columns = [col for col in data.columns if "ground_truth" in col or "epsilon" in col]
            data=data.drop(columns=removed_columns)
            output_n=neuro_combat_train(data)
            output_d=d_combat_train(data,path1)
            
            with open(os.path.join(path1,"output_n.pkl"),'wb') as f1:
                pickle.dump(output_n,f1)
            with open(os.path.join(path1,"output_d.pkl"),'wb') as f2:
                    pickle.dump(output_d,f2)
        except Exception as e:
            print(f"Error occurred: {e}")
            sys.exit(1)

else:
    data=pd.read_csv(os.path.join(results_common_path,"data.csv"))
    removed_columns = [col for col in data.columns if "ground_truth" in col or "epsilon" in col]
    data=data.drop(columns=removed_columns)

    output_n=neuro_combat_train(data)

    output_d=d_combat_train(data,results_common_path)

    with open(os.path.join(results_common_path,"output_n.pkl"),'wb') as f1:
        pickle.dump(output_n,f1)
    with open(os.path.join(results_common_path,"output_d.pkl"),'wb') as f2:
        pickle.dump(output_d,f2)
