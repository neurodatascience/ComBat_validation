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
#make plots such that for each row, I am presenting ground_truth, non-harmonized, neurocombat and d-combat
# data_path='/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/d-ComBat_project/simulated_data/data_Heterogeneity_N10000_G10_I4.csv'
script_dir=os.path.realpath(os.path.dirname(__file__))
sampling_type="Heterogeneity"
age_type="Homogeneous"
effect_type='linear'
N=1000
G=2
I=4
Data_path=os.path.join(script_dir,
                              "simulated_data")
data=pd.read_csv(os.path.join(Data_path,
                              f'data_{sampling_type}_age{age_type}_fixed{effect_type}_N{N}_G{G}_I{I}.csv'))
file_path=f'/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{sampling_type}_{age_type}_{effect_type}_N{N}_G{G}_I{I}'
filenames =os.listdir(file_path)#['site_out_1.pickle','site_out_2.pickle','site_out_3.pickle']
#os.listdir(file_path)
filenames=[f for f in filenames if "site_out" in f]
# print(filenames)
sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(sorted_filenames)

sites=[]
for filename in sorted_filenames:
    file_full_path = os.path.join(file_path, filename) 

    with open(file_full_path, "rb") as f:
        site_data = pickle.load(f)
        sites.append(site_data["dat_combat"])
d_combat = np.column_stack(sites).T
d_combat=pd.DataFrame(d_combat, columns=[f'd_c{i}' for i in range(G)])
# print(data.shape)
# print(d_combat.shape)

neuro_combat=pd.read_csv(f"/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{sampling_type}_{age_type}_{effect_type}_N{N}_G{G}_I{I}/neuro_data.csv")
neuro_combat=neuro_combat.T
neuro_combat.columns=[f'd_c{i}' for i in range(G)]

fig, axes = plt.subplots(G, 4, figsize=(15, 8))  # 10 rows, 4 columns

# Define color cycle
color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

unique_combinations = [(b, s) for b in data['batch'].unique() for s in np.unique(data['sex'])]
color_map = cm.get_cmap('tab10', len(unique_combinations))
color_dict = {combo: color_map(i) for i, combo in enumerate(unique_combinations)}


legend_entries = {}

for i in range(G):  # for 10 features
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

        j = batch - 1
        site_j = sites[j]
        y_c = site_j.iloc[i, :]  

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

        j = batch - 1
        site_j = sites[j]
        y_c = site_j.iloc[i, :]  

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
plt.show()
