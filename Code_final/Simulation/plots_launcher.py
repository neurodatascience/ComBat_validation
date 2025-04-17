import os
import subprocess
import json

#be careful of your working directory
script_dir=os.path.dirname(os.path.realpath(__file__))
script_path1 = os.path.join(script_dir,"gamma_delta_plot.py")
script_path2 = os.path.join(script_dir,"harmonized_data_plot.py")
default_path="/Users/xiaoqixie/Desktop/Mcgill/Winter_Rotation/combat_sites"

N_list=[5, 6, 8, 10] + list(range(12, 302, 20))

base_config_path = os.path.join(default_path,"test1",
                        "simulation_parameters.json")

with open(base_config_path, "r") as f:
    base_config = json.load(f)


for i in range(len(base_config)):
    config = base_config[i]  
    args = []
    for k, v in config.items():
        args.extend([f"--{k}", str(v)])

    subprocess.run(["python", script_path1] + args)   
    subprocess.run(["python", script_path2] + args) 