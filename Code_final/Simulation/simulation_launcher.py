import os
import subprocess
import json

script_dir=os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_dir,"Simulation.py")

base_config_path = os.path.join(script_dir,"simulation.json")

default_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites"

#****where need to modify

config={"store_folder":"test1",#folder name storing simulations
        "sampling_type": "H",
        "simulation_times": 100,
        "sex_type": "In",
        "age_type": "In",
        "effect_type": "nonlinear",
        "G": 100,
        "I": 5,
        "gamma_scale": 4}

numbers = [5, 6, 8, 10] + list(range(12, 302, 20))#size per batch
size_list = [num * config["I"] for num in numbers]#total sample size
#at least 5 is needed to avoid singular matrix through computation in combat models
##*****

################################################################
for i, size in enumerate(size_list):
    config["N"] = size
    # Save to temp config file
    with open (base_config_path, "w") as f:
        json.dump(config, f, indent=4)

    ####create a json file cotinuously adding simulation settings in it as rows#####
    parameter_folder_path = os.path.join(default_path, config["store_folder"])
    os.makedirs(parameter_folder_path, exist_ok=True)
    parameter_file_path = os.path.join(parameter_folder_path, "simulation_parameters.json")

    if not os.path.exists(parameter_file_path):
        with open(parameter_file_path, "w") as f:
            json.dump([config], f, indent=4)
    else:
        # If file exists, load it, append the config, and write back
        with open(parameter_file_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        data.append(config)
        with open(parameter_file_path, "w") as f:
            json.dump(data, f, indent=4)    

    print(f"Running simulation with N={size} (run {i+1}/{len(size_list)})")
    subprocess.run(["python", script_path, "--config", base_config_path])


