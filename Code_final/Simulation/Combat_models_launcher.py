import os
import subprocess
import json

script_dir=os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_dir,"Combat_models.py")

###specify your own working directory
default_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites"

base_config_path = os.path.join(default_path,"test1",
                        "simulation_parameters.json")
######

with open(base_config_path, "r") as f:
    base_config = json.load(f)


for i in range(len(base_config)):
    config = base_config[i]  
    args = []
    for k, v in config.items():
        args.extend([f"--{k}", str(v)])

    subprocess.run(["python", script_path] + args)    