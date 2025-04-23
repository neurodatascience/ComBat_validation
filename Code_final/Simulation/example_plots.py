# This script generates rmse and example plots of the data before and after harmonization following model training in example_models.py, 
# comparing them against the ground truth.

import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

default_path="/Users/xiaoqixie/Desktop/Mcgill/Winter_Rotation/combat_sites"

file_path=os.path.join(default_path,"test2","N200_0.5","simulation_0")

data=pd.read_csv(os.path.join(file_path,f"data.csv"))

with open(os.path.join(file_path,'output_d.pkl'),'rb') as f:
    d_output=pickle.load(f) 

keys=d_output.keys()
d_combat=[]
for k in keys:
    d_combat.append(d_output[k]['combat_data'])

d_combat=pd.concat(d_combat,axis=1).T

G=d_combat.shape[1]
print("G:",G)
d_combat.columns=[f'd_c{i}' for i in range(G)]
print("d_combat.shape:",d_combat)

file_path1=f"{file_path}/output_n.pkl"
with open(file_path1, "rb") as f:
    n_combat = pickle.load(f)

neuro_combat=pd.DataFrame(n_combat['combat_data'])
neuro_combat=neuro_combat.T
neuro_combat.columns=[f'd_c{i}' for i in range(G)]

#rmse
rmse_n1=np.square(np.mean((neuro_combat.iloc[:,0]-data["feature 0"])**2))
rmse_n2=np.square(np.mean((neuro_combat.iloc[:,1]-data["feature 1"])**2))
print("RMSE of feature 0 is (n-combat) ",rmse_n1)
print("RMSE of feature 1 is (n-combat)",rmse_n2)

rmse_d1=np.square(np.mean((d_combat.iloc[:,0]-data["feature 0"])**2))
rmse_d2=np.square(np.mean((d_combat.iloc[:,1]-data["feature 1"])**2))
print("RMSE of feature 0 is (d-combat)",rmse_d1)
print("RMSE of feature 1 is (d-combat)",rmse_d2)


fig, axes = plt.subplots(G, 4, figsize=(8*G, 10)) 

# Define color cycle
color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

unique_combinations = [(b, s) for b in data['batch'].unique() for s in np.unique(data['sex'])]
color_map = cm.get_cmap('tab10', len(unique_combinations))
color_dict = {combo: color_map(i) for i, combo in enumerate(unique_combinations)}


legend_entries = {}

for i in range(G): 
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    feature_name = f'feature {i}'
    ground_name = f'ground_truth {i}'

    # First pass to determine axis limits
    for batch in data['batch'].unique():
        s=np.where(data['batch']==batch)[0]
        d = data.iloc[s,]
        age = d['age']
        ground = d[ground_name]  
        y = d[feature_name]  
        
        y_n = neuro_combat.iloc[s, i]

        y_c = d_combat.iloc[s, i]  

        x_min = min(x_min, age.min())-5
        x_max = max(x_max, age.max())+5

        y_min = min(y_min, ground.min(), y.min(), y_n.min(), y_c.min())-2
        y_max = max(y_max, ground.max(), y.max(), y_n.max(), y_c.max())+2

    
    for batch in data['batch'].unique():
        # Second pass to plot
        s=np.where(data['batch']==batch)[0]
        d = data.iloc[s,]
        age = d['age']
        ground = d[ground_name]  
        y = d[feature_name]  
        current_sex = d['sex'].values  

        y_n = neuro_combat.iloc[s, i] 

        y_c = d_combat.iloc[s, i] 

        unique_sexes = np.unique(current_sex)

        for s in unique_sexes:
            indices = np.where(current_sex == s)[0]
            color = color_dict[(batch, s)]  # Get color from the dictionary

            row = i
            col = 0  

            y_min1=ground.iloc[indices].min()
            y_max1=ground.iloc[indices].max()
            # Ground-truth plot
            ax = axes[row, col]
            scatter = ax.scatter(age.iloc[indices], ground.iloc[indices], label=f'batch {batch}, sex {s}', s=8, color=color)
            ax.set_title(f'Ground Truth - {feature_name}')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axhline(y=y_min1, color='grey', linestyle='--', linewidth=1,alpha=0.2)
            ax.axhline(y=y_max1, color='grey', linestyle='--', linewidth=1,alpha=0.2)

            if f'batch {batch}, sex {s}' not in legend_entries:
                legend_entries[f'batch {batch}, sex {s}'] = scatter


            y_min2=y.iloc[indices].min()
            y_max2=y.iloc[indices].max()

            # Non-harmonized plot
            col = 1
            ax = axes[row, col]
            ax.scatter(age.iloc[indices], y.iloc[indices], s=8, color=color)
            ax.set_title(f'Non-Harmonized - {feature_name}')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axhline(y=y_min2, color='grey', linestyle='--', linewidth=1,alpha=0.2)
            ax.axhline(y=y_max2, color='grey', linestyle='--', linewidth=1,alpha=0.2)


            y_min3=y_n.iloc[indices].min()
            y_max3=y_n.iloc[indices].max()
            # Neuro-combat plot
            col = 2
            ax = axes[row, col]
            ax.scatter(age.iloc[indices], y_n.iloc[indices], s=8, color=color)
            ax.set_title(f'Neuro-Combat - {feature_name}')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axhline(y=y_min3, color='grey', linestyle='--', linewidth=1,alpha=0.2)
            ax.axhline(y=y_max3, color='grey', linestyle='--', linewidth=1,alpha=0.2)


            y_min4=y_c.iloc[indices].min()
            y_max4=y_c.iloc[indices].max()            
            # D-combat plot
            col = 3
            ax = axes[row, col]
            ax.scatter(age.iloc[indices], y_c.iloc[indices], s=8, color=color)
            ax.set_title(f'D-Combat - {feature_name}')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel("Age")
            ax.set_ylabel(feature_name)
            ax.axhline(y=y_min3, color='grey', linestyle='--', linewidth=1,alpha=0.2)
            ax.axhline(y=y_max3, color='grey', linestyle='--', linewidth=1,alpha=0.2)

fig.legend(handles=legend_entries.values(), labels=legend_entries.keys(),
           loc='upper left', bbox_to_anchor=(0.85, 1))

plt.tight_layout(rect=[0, 0, 0.85, 1])  
plt.savefig(os.path.join(file_path,"model_comparison.png"))
plt.close()
