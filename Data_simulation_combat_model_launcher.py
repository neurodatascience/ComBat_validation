import os
import subprocess
import json

script_dir=os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_dir,"Data_simulation_ComBat_model.py")

default_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites"

N_list=[2,4,6,8,10,12,14,16,18,20,22,24,26,32,48,60,72,84,90,100,120,140,160,180,200,220,240,260,280,300]

for i, N in enumerate(N_list):
    base_config_path = os.path.join(default_path,f"min_points{N}",
                            f"Homogeneous_Heterogenity_Heterogenity_nonlinear_N{N*5}_G100_I5_Gamma4",
                            "simulation.json")

    print(f"Running simulation with N={N} (run {i+1}/{len(N_list)})")
    subprocess.run(["python", script_path, "--config", base_config_path])


