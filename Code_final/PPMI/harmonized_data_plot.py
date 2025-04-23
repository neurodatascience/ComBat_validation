#This script is used to plotnon-haronized data with harmonized data from neural-combat and distributed-combat models.
import os
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/winter_rotation/Code_final")
from helper import bootstrap_ntimes,neuro_combat_train,d_combat_train

import matplotlib.pyplot as plt

common_path="/Users/xiaoqixie/Desktop/Mcgill/Winter_Rotation"
ppmi_case_folder_path=os.path.join(common_path,"PPMI")

data=pd.read_csv(os.path.join(ppmi_case_folder_path,"data_at_least6.csv"))

path1=os.path.join(ppmi_case_folder_path,"combat_outputs")
with open(os.path.join(path1,"d_output.pkl"),'rb') as f:
    d_output=pickle.load(f)

key=d_output.keys()
d_combat=[]
for k in key:
    d_combat.append(d_output[k]['combat_data'])
d_combat=pd.concat(d_combat,axis=1)
# print(d_combat.shape)
d_combat=d_combat.T

with open(os.path.join(path1,"n_output.pkl"),'rb') as f:
    n_output=pickle.load(f)
n_combat=pd.DataFrame(n_output['combat_data'].T)
###########################################################################
feature=data.drop(columns=['age','sex','batch'])
feature_names=feature.columns
G = len(feature_names)

legend_entries = {}

def plot_scatter(ax, x, y, color, title, xlim, ylim):
    ax.scatter(x, y, s=8, color=color)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.axhline(y=y.min(), color='grey', linestyle='--', linewidth=1, alpha=0.2)
    # ax.axhline(y=y.max(), color='grey', linestyle='--', linewidth=1, alpha=0.2)

unique_combinations = [(b, s) for b in data['batch'].unique() for s in np.unique(data['sex'])]
color_map = plt.colormaps.get_cmap('tab20').resampled(len(unique_combinations))
color_dict = {combo: color_map(i) for i, combo in enumerate(unique_combinations)}

for i in range(G): 
    feature_name = feature_names[i]
    legend_entries = {}

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    for batch in data['batch'].unique():
        s = np.where(data['batch'] == batch)[0]
        d = data.iloc[s]
        age = d['age']
        y = d[feature_name]  
        y_n = n_combat.iloc[s, i]
        y_d = d_combat.iloc[s, i]
        x_min = min(x_min, age.min()) - 5
        x_max = max(x_max, age.max()) + 5
        y_min = min(y_min, y.min(), y_n.min(), y_d.min()) - 2
        y_max = max(y_max, y.max(), y_n.max(), y_d.max()) + 2

    for batch in data['batch'].unique():
        s = np.where(data['batch'] == batch)[0]
        d = data.iloc[s]
        age = d['age']
        y = d[feature_name]
        current_sex = d['sex'].values
        y_n = n_combat.iloc[s, i]
        y_d = d_combat.iloc[s, i]

        for s_val in np.unique(current_sex):
            indices = np.where(current_sex == s_val)[0]
            color = color_dict[(batch, s_val)]

            # Non-Harmonized
            plot_scatter(axes[0], age.iloc[indices], y.iloc[indices], color,
                         'Non-Harmonized', (x_min, x_max), (y_min, y_max))
            
            if f'batch {batch}, sex {s_val}' not in legend_entries:
                legend_entries[f'batch {batch}, sex {s_val}'] = axes[0].scatter([], [], color=color)

            # Neuro-ComBat
            plot_scatter(axes[1], age.iloc[indices], y_n.iloc[indices], color,
                         'Neuro-ComBat', (x_min, x_max), (y_min, y_max))

            # Distributed-ComBat
            plot_scatter(axes[2], age.iloc[indices], y_d.iloc[indices], color,
                         'D-ComBat', (x_min, x_max), (y_min, y_max))

    fig.suptitle(f'Feature: {feature_name}', fontsize=16)
    fig.legend(
        handles=legend_entries.values(),
        labels=legend_entries.keys(),
        loc='upper right',
        bbox_to_anchor=(1, 1),
        borderaxespad=0.0,
        title="Batch & Sex",
        ncol=3
    )
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    # Save each figure
    plt.savefig(os.path.join(ppmi_case_folder_path, f"{feature_name}.png"), dpi=300)
    plt.close()