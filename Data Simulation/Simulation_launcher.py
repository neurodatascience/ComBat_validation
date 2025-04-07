import os
import subprocess
import json

script_dir=os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_dir,"Simulation.py")

base_config_path = os.path.join(script_dir,"simulation.json")
with open(base_config_path, "r") as f:
    base_config = json.load(f)

N_list=list(range(2, 300, 2))
for i, N in enumerate(N_list):
    config = base_config.copy()
    config["N"] = N*5
    print(N)
    # Save to temp config file
    with open (base_config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Running simulation with N={N} (run {i+1}/{len(N_list)})")
    subprocess.run(["python", script_path, "--config", base_config_path])

