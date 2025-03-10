import os 
import matplotlib.pyplot as plt
script_dir=os.getcwd()
import pickle
import numpy as np
import pandas as pd
"""Global setup"""
sampling_type="Heterogeneity"
age_type="Homogeneous"
effect_type='linear'
N=1000
G=2
I=4
#import two models
file_path=f'/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{sampling_type}_{age_type}_{effect_type}_N{N}_G{G}_I{I}'
print("==================================================================================================")
print("import true data")
Data_path=os.path.join(script_dir,"simulated_data")
Data=pd.read_csv(os.path.join(Data_path,
                              f'data_{sampling_type}_age{age_type}_fixed{effect_type}_N{N}_G{G}_I{I}.csv'))
print("=========================================================================================================")
print("ground truth data")
y_columns=[name for name in Data.columns if "ground_truth" in name]
y_ground=Data[y_columns]
print("====================================================================================================")
print("import d-combat data")
filenames =os.listdir(file_path)
filenames=[f for f in filenames if "site_out" in f]
sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(sorted_filenames)
sites=[]
for i in range(I):
    file_path2=os.path.join(file_path,sorted_filenames[i])
    with open(file_path2,"rb") as f:
        sites.append(pickle.load(f))
# print(sites[0])
#get gamma_star and delta_star
gamma_star_d=[]
delta_star_d=[]
var_pooled_d=[]
for i in range(I):
    g=sites[i]['estimates']['gamma_star']
    d=sites[i]['estimates']['delta_star']
    v=sites[i]['estimates']['var_pooled']
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
file_path1=f"{file_path}/neuro_combat.pickle"
with open(file_path1, "rb") as f:
    n_combat = pickle.load(f)


delta_star_n=pd.DataFrame(n_combat['estimates']['delta.star'])#four sites, each site has two features
gamma_star_n=pd.DataFrame(n_combat['estimates']['gamma.star'])
var_pooled_n=pd.DataFrame(n_combat['estimates']['var.pooled'])
print("delta_star_n:",delta_star_n)
print("gamma_star_n:",gamma_star_n)
print("var_pooled_n:",var_pooled_n)
y_combat=n_combat["data"]
print("======================================================================================================")
print("import n_samples")
n_samples=pd.read_csv(os.path.join(file_path,'n_samples.csv'))
n_samples=n_samples.to_numpy()
print("======================================================================================================")
print("import neuro combat data")
file_path1=f"{file_path}/neuro_combat.pickle"
with open(file_path1, "rb") as f:
    n_combat = pickle.load(f)

print(n_combat)
delta_star_n=pd.DataFrame(n_combat['estimates']['delta.star'])#four sites, each site has two features
gamma_star_n=pd.DataFrame(n_combat['estimates']['gamma.star'])
print("delta_star_n:",delta_star_n)
print("gamma_star_n:",gamma_star_n)
y_combat=n_combat["data"]
print("======================================================================================================")
print("import n_samples")
n_samples=pd.read_csv(os.path.join(file_path,'n_samples.csv'))
n_samples=n_samples.to_numpy()
print("======================================================================")
print("import true gamma and delta")
gamma_IG=pd.read_csv(os.path.join(file_path,'gamma_IG.csv')).T
delta_IG=pd.read_csv(os.path.join(file_path,'delta_IG.csv')).T
# print(gamma_IG.shape)
print("==============================================")
print("plot gamma estimator and delta estimator")
colors = {
    "gamma_ig": "blue",
    "gamma_star_n_ig": "red",
    "gamma_star_d_ig": "black",
    "delta_ig": "blue",
    "delta_star_n_ig":"red",
    "delta_star_d_ig": "black",
}

fig, axes = plt.subplots(I, G, figsize=(G * 3, I * 3), constrained_layout=True)

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
        y = [gamma_ig, gamma_star_n_ig, gamma_star_d_ig, delta_ig, delta_star_n_ig, delta_star_d_ig]
        labels = ["gamma_ig", "gamma_star_n_ig", "gamma_star_d_ig", "delta_ig", "delta_star_n_ig", "delta_star_d_ig"]
        
        for xi, yi, label in zip(x, y, labels):
            ax.scatter(xi, yi, color=colors[label], label=label if i == 0 and g == 0 else "")
        
        ax.set_xticks([0.3, 0.8])
        ax.set_xticklabels(["gamma", "delta"], rotation=45)
        ax.set_title(f"i={i}, g={g}")

# Add legend outside
fig.legend(labels, loc="upper right",bbox_to_anchor=(1.3, 1))
plt.show()

print("=======================================================================")
print("gamma_ig+delta_ig*epsilon_ijg")
import numpy as np
import matplotlib.pyplot as plt

unique_sexes = Data['sex'].unique()
v_types = ['gamma+delta*epsilon', 'neuro gamma+delta*epsilon', 'd-com gamma+delta*epsilon']

cmap = plt.get_cmap("tab20")
color_map = {(sex, v): cmap(i / (len(unique_sexes) * len(v_types))) 
             for i, (sex, v) in enumerate([(s, v) for s in unique_sexes for v in v_types])}

fig, axes = plt.subplots(I, G, figsize=(G * 4, I * 4), sharex=True, sharey=True)
axes = np.array(axes)  

handles = []
labels = set() 
for i in range(I):
    for g in range(G):
        d = Data[Data['batch'] == (i + 1)].reset_index(drop=True)
        
        # Extract required columns
        epsilon_ig = d[f'epsilon {g}'].values
        age = d['age'].values
        
        # Compute V1, V2, V3
        gamma_ig = gamma_IG.iloc[i, g]
        gamma_star_n_ig = gamma_star_n.iloc[i, g] * (var_pooled_n.loc[g] ** 0.5)
        gamma_star_d_ig = gamma_star_d.iloc[i, g] * (var_pooled_d.iloc[i, g] ** 0.5)
        delta_ig = delta_IG.iloc[i, g]
        delta_star_n_ig = delta_star_n.iloc[i, g]
        delta_star_d_ig = delta_star_d.iloc[i, g]

        v1 = gamma_ig + delta_ig * epsilon_ig
        v2 = gamma_star_n_ig.values + delta_star_n_ig * epsilon_ig
        v3 = gamma_star_d_ig + delta_star_d_ig * epsilon_ig

        ax = axes[i, g]  # Get correct subplot

        # Plot each sex separately with unique color per (sex, V)
        for sex in unique_sexes:
            mask = (d['sex'] == sex).values  # Convert to NumPy array for faster indexing
            
            if np.any(mask):  # Only plot if there are valid points
                p1= ax.scatter(age[mask], v1[mask], color=color_map[(sex, 'gamma+delta*epsilon')], 
                                 alpha=1, s=10, label=f'gamma+delta*epsilon - {sex}' if (sex, 'gamma+delta*epsilon') not in labels else "")
                p2 = ax.scatter(age[mask], v2[mask], color=color_map[(sex, 'neuro gamma+delta*epsilon')], 
                                 alpha=1, s=20, label=f'neuro gamma+delta*epsilon - {sex}' if (sex, 'neuro gamma+delta*epsilon') not in labels else "")
                p3 = ax.scatter(age[mask], v3[mask], color=color_map[(sex, 'd-com gamma+delta*epsilon')], 
                                 alpha=1, s=30, label=f'd-com gamma+delta*epsilon - {sex}' if (sex, 'd-com gamma+delta*epsilon') not in labels else "")

                # Add unique handles and labels
                if (sex, 'gamma+delta*epsilon') not in labels:
                    handles.append(p1)
                    labels.add((sex, 'gamma+delta*epsilon'))
                if (sex, 'neuro gamma+delta*epsilon') not in labels:
                    handles.append(p2)
                    labels.add((sex, 'neuro gamma+delta*epsilon'))
                if (sex, 'd-com gamma+delta*epsilon') not in labels:
                    handles.append(p3)
                    labels.add((sex, 'd-com gamma+delta*epsilon'))

        ax.set_title(f'Batch {i+1}, G {g}')
        ax.set_xlabel('Age')
        ax.set_ylabel('Values')

# Adjust layout
plt.tight_layout()
fig.legend(handles, [h.get_label() for h in handles], loc='upper right', bbox_to_anchor=(1.4, 1))
plt.suptitle('Comparison of V1, V2, and V3 by Sex and Age', fontsize=14, y=1.02)
plt.show()
print("============================================================")

print("delta/delta_star")
for i in range(I):
    for g in range(G):
        delta_ig = delta_IG.iloc[i, g]
        delta_star_n_ig = delta_star_n.iloc[i, g]
        delta_star_d_ig = delta_star_d.iloc[i, g]
        print(f"neuro combat site{i} feature{g}:",delta_ig/delta_star_n_ig)
        print(f"d-combat site{i} feature{g}:",delta_ig/delta_star_d_ig)
print("estmated delta is much bigger than true delta") 
"""epsilon in y-combat is smaller than epsilon in y as the delta is canceled by over delta_star"""
