import os 
import matplotlib.pyplot as plt
script_dir=os.getcwd()
import pickle
import numpy as np
import pandas as pd
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
file_path=os.path.join(default_path,
                                           f"min_points{smallest_sample_size}",
                                            f'{file_name}')
print("==================================================================================================")
print("import true data")
Data=pd.read_csv(os.path.join(file_path,
                              f'{file_name}.csv'))

print("=========================================================================================================")
print("ground truth data")
y_columns=[name for name in Data.columns if "ground_truth" in name]
y_ground=Data[y_columns]
print("====================================================================================================")
print("import d-combat data")
with open(os.path.join(file_path,'output_d.pkl'),'rb') as f:
    d_output=pickle.load(f) 
#get gamma_star and delta_star
keys=d_output.keys()
gamma_star_d=[]
delta_star_d=[]
var_pooled_d=[]
for i,k in enumerate(keys):
    g=d_output[k]['gamma_star']
    d=d_output[k]['delta_star']
    v=d_output[k]['sigma']
    gamma_star_d.append(g)
    delta_star_d.append(d)
    var_pooled_d.append(v)
gamma_star_d=pd.DataFrame(np.column_stack(gamma_star_d)).T
delta_star_d=pd.DataFrame(np.column_stack(delta_star_d)).T
var_pooled_d=pd.DataFrame(np.column_stack(var_pooled_d)).T

print("gamma_star_d:",gamma_star_d)
print("delta_star_d:",delta_star_d)
print("var_pooled_d:",var_pooled_d)

print("======================================================================================================")
print("import neuro combat data")
file_path1=f"{file_path}/output_n.pkl"
with open(file_path1, "rb") as f:
    n_combat = pickle.load(f)


delta_star_n=pd.DataFrame(n_combat['delta_star'])#four sites, each site has two features
gamma_star_n=pd.DataFrame(n_combat['gamma_star'])
var_pooled_n=pd.DataFrame(n_combat['sigma'])
print("delta_star_n:",delta_star_n)
print("gamma_star_n:",gamma_star_n)
print("var_pooled_n:",var_pooled_n)
y_combat=n_combat["combat_data"]
print("======================================================================================================")
print("import n_samples")
n_samples=pd.read_csv(os.path.join(file_path,'n_samples.csv'))
n_samples=n_samples.to_numpy()
print("======================================================================================================")
print("import true gamma and delta")
gamma_IG=pd.read_csv(os.path.join(file_path,'gamma_IG.csv')).T
delta_IG=pd.read_csv(os.path.join(file_path,'delta_IG.csv')).T
# print(gamma_IG.shape)
print("==============================================")
print("plot gamma estimator and delta estimator")
colors = {
    "gamma_ig": "#1f77b4",          # blue
    "gamma_star_n_ig": "#d62728",   # red
    "gamma_star_d_ig": "#2ca02c",   # green
    "delta_ig": "#1f77b4",          # purple
    "delta_star_n_ig":"#d62728",   # orange
    "delta_star_d_ig":"#2ca02c",   # cyan
}

fig, axes = plt.subplots(I, G, figsize=(G * 3+10, I * 3), constrained_layout=False)

for i in range(I):
    for g in range(G):
        gamma_ig = gamma_IG.iloc[i, g]
        gamma_star_n_ig = gamma_star_n.iloc[i, g]*(var_pooled_n.loc[g]**0.5)
        gamma_star_d_ig = gamma_star_d.iloc[i, g]*(var_pooled_d.iloc[i,g]**0.5)
        delta_ig = delta_IG.iloc[i, g]
        delta_star_n_ig = delta_star_n.iloc[i, g]
        delta_star_d_ig = delta_star_d.iloc[i, g]
        
        ax = axes[i, g] if I > 1 and G > 1 else axes[max(i, g)]
        
        # Plot values as points
        x = [0.3, 0.3, 0.3, 0.8, 0.8, 0.8]  # Align gamma values vertically, same for delta
        y = [gamma_star_n_ig, gamma_star_d_ig,gamma_ig, delta_star_n_ig, delta_star_d_ig,delta_ig]
        labels = [ "gamma_star_n_ig", "gamma_star_d_ig","gamma_ig", "delta_star_n_ig", "delta_star_d_ig","delta_ig"]
        
        for xi, yi, label in zip(x, y, labels):
            ax.scatter(
                xi, yi, 
                color=colors[label], 
                label=label if i == 0 and g == 0 else "",
                alpha=0.6,  # Add transparency for blending
                edgecolors='w',  # Optional: white edges make points more distinguishable
                linewidths=0.5,
                s=40
            )
        
        ax.set_xticks([0.3, 0.8])
        ax.set_xticklabels(["gamma", "delta"], rotation=45)
        ax.set_title(f"batch={i}, feature={g}")

# Add legend outside
fig.legend(labels, loc="upper right",bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(file_path,"gamma_delta.png"))
plt.close()
print("============================================================")
