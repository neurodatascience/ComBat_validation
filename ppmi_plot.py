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
print("====================================================================================")
print("import resampled data")
script_dir=os.path.realpath(os.path.dirname(__file__))

folder_name="simulated_data"
Data_path=os.path.join(script_dir,folder_name)
file_name=f'data_gamma0.5_nonlinear'
data=pd.read_csv(os.path.join(Data_path,f"{file_name}.csv"))

if folder_name=="simulated_data":
    col = [name for name in data.columns if "ground" in name or "epsilon" in name]
    data = data.drop(columns=col)
print("data.columns:",data.columns)

batch_id=data["batch"].drop_duplicates().tolist()
print("batch_id:",batch_id)
data2=data.drop(columns=["age","sex","batch"])
feature_name=data2.columns
unique_batch=data['batch'].unique()
I=len(unique_batch)#number of sites
G=len(feature_name)#number of features
print("number of sites:",I)
print("number of features:",G)
print("===================================================================================")
print("import d-combat data")
file_path=f'/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{file_name}'
filenames =os.listdir(file_path)
filenames=[f for f in filenames if "site_out" in f]
sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(sorted_filenames)

sites=[]
for filename in sorted_filenames:
    file_full_path = os.path.join(file_path, filename) 

    with open(file_full_path, "rb") as f:
        site_data = pickle.load(f)
        sites.append(site_data["dat_combat"])
d_combat = pd.DataFrame(np.column_stack(sites).T).reset_index(drop=True)
# d_combat=pd.DataFrame(d_combat, columns=[f'd_c{i}' for i in range(G)])
print("d_combat.shape:",d_combat.shape)
#####################################################################################
gamma_star_d=[]
delta_star_d=[]
var_pooled_d=[]
# alpha_hat_d=[]
phi_hat_d=[]

sites1=[]
for filename in sorted_filenames:
    file_full_path = os.path.join(file_path, filename) 

    with open(file_full_path, "rb") as f:
        site_data = pickle.load(f)
        sites1.append(site_data)

for i in range(I):
    g=sites1[i]['estimates']['gamma_star']
    d=sites1[i]['estimates']['delta_star']
    v=sites1[i]['estimates']['var_pooled']
    p=sites1[i]['estimates']['mod_mean']
    gamma_star_d.append(g)
    delta_star_d.append(d)
    var_pooled_d.append(v)
    phi_hat_d.append(p)

gamma_star_d=pd.DataFrame(np.column_stack(gamma_star_d)).T.reset_index(drop=True)
delta_star_d=pd.DataFrame(np.column_stack(delta_star_d)).T.reset_index(drop=True)
var_pooled_d=pd.DataFrame(np.column_stack(var_pooled_d)).T.reset_index(drop=True)
alpha_hat_d=np.unique(sites1[0]['estimates']['stand_mean'],axis=1).tolist()
phi_hat_d=pd.DataFrame(np.column_stack(phi_hat_d)).T.reset_index(drop=True)

print("gamma_star_d.shape:",gamma_star_d.shape)
print("delta_star_d.shape:",delta_star_d.shape)
print("var_pooled_d.shape:",var_pooled_d.shape)
print("len(alpha_hat_d):",len(alpha_hat_d))
print("phi_hat_d.shape:",phi_hat_d.shape)
print("=====================================================================================")
print("import neuro-combat data")
neuro_combat=pd.read_csv(f"/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{file_name}/neuro_data.csv")
neuro_combat=neuro_combat.T
print("neuro_combat.shape:",neuro_combat.shape)

file_path1=f"{file_path}/neuro_combat.pickle"
with open(file_path1, "rb") as f:
    n_combat = pickle.load(f)


delta_star_n=pd.DataFrame(n_combat['estimates']['delta.star']).reset_index(drop=True)
gamma_star_n=pd.DataFrame(n_combat['estimates']['gamma.star']).reset_index(drop=True)
var_pooled_n=pd.DataFrame(n_combat['estimates']['var.pooled']).reset_index(drop=True)
alpha_hat_n=np.unique(n_combat['estimates']['stand.mean'],axis=1).tolist()
phi_hat_n=pd.DataFrame(n_combat['estimates']['mod.mean']).T.reset_index(drop=True)

print("gamma_star_n.shape:",gamma_star_n.shape)
print("delta_star_n.shape:",delta_star_n.shape)
print("var_pooled_n.shape:",var_pooled_n.shape)
print("len(alpha_hat_n):",len(alpha_hat_n))
print("phi_hat_n.shape:",phi_hat_n.shape)
print("=================================================================================")
# print("plot non-harmonized and harmonized data")
# fig, axes = plt.subplots(G, 3, figsize=(30,10*G)) #non-harmonized, combat and d-combat

# color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# unique_combinations = [(b, s) for b in data['batch'].unique() for s in np.unique(data['sex'])]
# color_map = cm.get_cmap('tab10', len(unique_combinations))
# color_dict = {combo: color_map(i) for i, combo in enumerate(unique_combinations)}


# legend_entries = {}
# # interested_batch=[15,26]
# for i in range(G):  
#     x_min, x_max = float('inf'), float('-inf')
#     y_min, y_max = float('inf'), float('-inf')


#     j=0
#     for batch in data['batch'].unique():
#         s=np.where(data['batch']==batch)[0]
#         d = data.iloc[s,]
#         age = d['age']

#         y = d[feature_name[i]]  
        
#         y_n = neuro_combat.iloc[s, i]

#         site_j = sites[j]
#         j+=1
#         y_c = site_j.iloc[i, :]  

#         x_min = min(x_min, age.min())-5
#         x_max = max(x_max, age.max())+5

#         y_min = min(y_min, y.min(), y_n.min(), y_c.min())-2
#         y_max = max(y_max, y.max(), y_n.max(), y_c.max())+2

#     j=0
#     for batch in data['batch'].unique():
        
#         s=np.where(data['batch']==batch)[0]
#         d = data.iloc[s,]
#         age = d['age']
#         y = d[feature_name[i]]  
#         current_sex = d['sex'].values  

#         y_n = neuro_combat.iloc[s, i] 

#         site_j = sites[j]
#         j+=1
#         y_c = site_j.iloc[i, :]  

#         unique_sexes = np.unique(current_sex)

#         for s in unique_sexes:
#             indices = np.where(current_sex == s)[0]
#             color = color_dict[(batch, s)]  

#             row = i
#             col = 0  

#             y_min2=y.iloc[indices].min()
#             y_max2=y.iloc[indices].max()

#             # Non-harmonized plot
#             col = 0
#             ax = axes[row, col]
#             ax.scatter(age.iloc[indices], y.iloc[indices], s=15, color=color)
#             ax.set_title(f'Non-Harmonized - {feature_name[i]}')
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             scatter = ax.scatter(age.iloc[indices], y.iloc[indices], label=f'batch {batch}, sex {s}', s=15, color=color)
#             if f'batch {batch}, sex {s}' not in legend_entries:
#                 legend_entries[f'batch {batch}, sex {s}'] = scatter


#             y_min3=y_n.iloc[indices].min()
#             y_max3=y_n.iloc[indices].max()
#             # Neuro-combat plot
#             col = 1
#             ax = axes[row, col]
#             ax.scatter(age.iloc[indices], y_n.iloc[indices], s=15, color=color)
#             ax.set_title(f'Neuro-Combat - {feature_name[i]}')
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)


#             y_min4=y_c.iloc[indices].min()
#             y_max4=y_c.iloc[indices].max()            
#             # D-combat plot
#             col = 2
#             ax = axes[row, col]
#             ax.scatter(age.iloc[indices], y_c.iloc[indices], s=15, color=color)
#             ax.set_title(f'D-Combat - {feature_name[i]}')
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             ax.set_xlabel("Age")
#             ax.set_ylabel(feature_name[i])

# fig.legend(handles=legend_entries.values(), labels=legend_entries.keys(),
#            loc='upper left', bbox_to_anchor=(0.85, 1))

# plt.tight_layout(rect=[0, 0, 0.85, 1])  
# plt.savefig(os.path.join(file_path,"model_comparison.png"))
# plt.show()

print("====================================================================================")
# print("plot gamma start and delta star")

# colors = {
#     "gamma_star_n_ig": "red",
#     "gamma_star_d_ig": "black",

#     "delta_star_n_ig":"red",
#     "delta_star_d_ig": "black",
# }

# fig, axes = plt.subplots(I, G, figsize=(G * 3+10, I * 3), constrained_layout=False)
# # print(delta_star_n)
# for i in range(I):
#     for g in range(G):
#         gamma_star_n_ig = gamma_star_n.iloc[i, g]
#         gamma_star_d_ig = gamma_star_d.iloc[i, g]
       
#         delta_star_n_ig = delta_star_n.iloc[i, g]
#         delta_star_d_ig = delta_star_d.iloc[i, g]
        
#         ax = axes[i, g] if I > 1 and G > 1 else axes[max(i, g)]
        
#         # Plot values as points
#         x = [0.3, 0.3, 0.8, 0.8]  # Align gamma values vertically, same for delta
#         y = [gamma_star_n_ig, gamma_star_d_ig, delta_star_n_ig, delta_star_d_ig]
#         labels = ["gamma_star_n_ig", "gamma_star_d_ig", "delta_star_n_ig", "delta_star_d_ig"]
        
#         for xi, yi, label in zip(x, y, labels):
#             ax.scatter(xi, yi, color=colors[label], label=label if i == 0 and g == 0 else "")
        
#         ax.set_xticks([0.3, 0.8])
#         ax.set_xticklabels(["gamma", "delta"], rotation=45)
#         ax.set_title(f"batch={batch_id[i]}, feature={g}")

# # Add legend outside
# fig.legend(labels, loc="upper right",bbox_to_anchor=(1, 1))
# plt.savefig(os.path.join(file_path,"gamma_delta.png"))
# plt.show()

print("====================================================================================")
print("plot residual following N(0,sigma^2)")

fig, axes = plt.subplots(G, 1, figsize=(1.8*I,G * 3+20), constrained_layout=True)
fig.subplots_adjust(right=0.6)
for g in range(G):   
    age = []
    res_d = []
    res_n = []

    for i in range(I):
        s = np.where(data['batch'] == unique_batch[i])[0]
        d = data.iloc[s, :]
        
        age.append(d['age'])  
        
        y_d = d_combat.iloc[s, g]
        y_n = neuro_combat.iloc[s, g]  
        
        print("len(y_d) == len(y_n):", len(y_d) == len(y_n))

        e_d = y_d - np.repeat(alpha_hat_d[g], len(y_d)) - phi_hat_d.iloc[s, g].to_numpy()
        e_n = y_n - np.repeat(alpha_hat_n[g], len(y_n)) - phi_hat_n.iloc[s, g].to_numpy()
        print("phi_hat_d.iloc[s, g].to_numpy():",len(phi_hat_d.iloc[s, g].to_numpy()))
        print("phi_hat_n.iloc[s, g].to_numpy():",len(phi_hat_n.iloc[s, g].to_numpy()))
        res_d.append(pd.Series(e_d))
        res_n.append(pd.Series(e_n))

    res_d = pd.concat(res_d, axis=0).reset_index(drop=True).values
    res_n = pd.concat(res_n, axis=0).reset_index(drop=True).values
    age = pd.concat(age, axis=0).reset_index(drop=True).values

    print("Final age shape:", age.shape)
    print("res_n:",len(res_n))
    print("Plotting residuals for group", g)
    axes[g].scatter(age, res_n, label=f"Residuals (n-combat data) batch", alpha=0.5)
    axes[g].scatter(age, res_d, label=f"Residuals (d-combat data) batch", alpha=0.5)
    axes[g].axhline(y=0, color='red', linestyle='--')
    axes[g].set_xlabel("Age")
    axes[g].set_ylabel("Residuals")
    axes[g].legend(loc="upper right",bbox_to_anchor=(1.3, 1))
    axes[g].set_title(f"Residuals for feature {feature_name[g]}")
 

save_path=os.path.join("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites",
                       file_name,"residuals.png")
plt.savefig(save_path)
plt.show()

print("==============================================================================")
# print("plot alpha")
# plt.figure(figsize=(14,12))  
# plt.scatter(feature_name, alpha_hat_n, color='red', label="neuro-combat alpha")
# plt.scatter(feature_name, alpha_hat_d, color='blue', label="d-combat alpha")
# plt.xlabel("Features")
# plt.ylabel("Alpha values")
# plt.legend()
# plt.xticks(rotation=45)  
# plt.title("Comparison of Alpha Values for Neuro-Combat and D-Combat")
# save_path=os.path.join("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites",
#                        file_name,"alpha_hat.png")
# plt.savefig(save_path)
# plt.show()
# print("=======================================================================================")
# print("plot phi for each feature")

# colors = {('d-combat', 0): 'red', ('d-combat', 1): 'blue',
#           ('n-combat', 0): 'orange', ('n-combat', 1): 'green'}

# fig, axes = plt.subplots(G, 1, figsize=(1.8*I, G * 3 + 20), constrained_layout=True)
# fig.subplots_adjust(right=0.8)

# for g in range(G):   
#     age = data['age'].to_numpy()
#     sex = data['sex'].to_numpy()


#     for s in [0, 1]:
#         mask = (sex == s)  
#         axes[g].scatter(age[mask], phi_hat_d.iloc[mask, g], 
#                         color=colors[('d-combat', s)], label=f"d-combat (sex {s})", alpha=0.5)

#     for s in [0, 1]:
#         mask = (sex == s)
#         axes[g].scatter(age[mask], phi_hat_n.iloc[mask, g].to_numpy(), 
#                         color=colors[('n-combat', s)], label=f"n-combat (sex {s})", alpha=0.5)


#     axes[g].axhline(y=0, color='black', linestyle='--')

#     axes[g].set_xlabel("Age")
#     axes[g].set_ylabel("Fixed effects")
#     axes[g].set_title(f"Fixed effects for feature {feature_name[g]}")

#     axes[g].legend(loc="upper left", bbox_to_anchor=(1.1, 1))

# save_path=os.path.join("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites",
#                        file_name,"fixed_effect.png")
# plt.savefig(save_path)
# plt.show()
# print("===========================================================================================")
# print("plot of N(0,1)")
# np.random.seed(42)  
# random_numbers = np.random.normal(0, 1, 1000)

# plt.figure(figsize=(8,6))

# plt.scatter(range(len(random_numbers)), random_numbers, 
#             facecolors='skyblue', edgecolors='black', alpha=0.7)

# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.title("Scatter Plot of Random Numbers from N(0,1)")

# plt.show()