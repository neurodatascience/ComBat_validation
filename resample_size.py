import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"

print("Import data data_80batches")
data_file="ppmi-age-sex-case-aseg"
Data_path=os.path.join(common_path,"d-ComBat_project",data_file)
file_name=f'data_80batches'
Data=pd.read_csv(os.path.join(Data_path,f'{file_name}.csv'))

print("===================================================================================")
print("group data by ids")
group=Data.groupby("batch").size().reset_index(name="sample_size")
top_10_batches = group.nlargest(10, "sample_size")
print(top_10_batches["sample_size"].sum())
print("============================================================================")
Data_sub=Data[Data["batch"].isin(top_10_batches["batch"])]
print(Data_sub.max(axis=0))
print(Data_sub.shape)
print("=============================================================")
print("how harmonized data from top 10 batches loos like")

print("will the harmonized data look different if we look at the same sites, "
"but one from estimations with 80 batches and one from with only top 10 batches")

print("import d-combat sites")
def import_d_combat(file_name):

    if file_name=='data_top10batches':

        sites_path=os.path.join(common_path,"combat_sites",data_file,file_name)

        filenames = [name for name in os.listdir(sites_path) if "site_out" in name]
        print(filenames)

        sites = {}  

        for filename in filenames:
            file_dir = os.path.join(sites_path, filename)  
            number=filename.split("_")[-1].split(".")[0]
            print(number)
            # Open pickle file
            with open(file_dir, "rb") as f:
                site_data = pickle.load(f)
                sites[number] = site_data["dat_combat"]  
    elif file_name=='data_80batches':
        filenames = [name for name in os.listdir(
            os.path.join(common_path,"combat_sites",data_file,
                                                            'data_top10batches')) 
                    if "site_out" in name]
        sites_path=os.path.join(common_path,"combat_sites",data_file,file_name)
        
        sites = {}  
        for filename in filenames:
            file_dir = os.path.join(sites_path, filename)  
            number=filename.split("_")[-1].split(".")[0]
            print(number)
            # Open pickle file
            with open(file_dir, "rb") as f:
                site_data = pickle.load(f)
                sites[number] = site_data["dat_combat"]  
    return(sites)

d_combat_80 = pd.DataFrame(np.hstack(list(import_d_combat("data_80batches").values())).T)
d_combat_10 = pd.DataFrame(np.hstack(list(import_d_combat("data_top10batches").values())).T)
print("d_combat_80.shape:",d_combat_80.shape)
print("d_combat_10.shape:",d_combat_10.shape)
print("==================================================================")
sites_path=os.path.join(common_path,"combat_sites",data_file,"data_80batches")
neuro_combat_80=pd.read_csv(f"{sites_path}/neuro_data.csv")
neuro_combat_80=neuro_combat_80.T
s=np.where(Data['batch'].isin(top_10_batches["batch"]))[0]
neuro_combat_80=neuro_combat_80.iloc[s,:]
print("neuro_combat_80batches.shape:",neuro_combat_80.shape)

sites_path=os.path.join(common_path,"combat_sites",data_file,"data_top10batches")
neuro_combat_10=pd.read_csv(f"{sites_path}/neuro_data.csv")
neuro_combat_10=neuro_combat_10.T
print("neuro_combat_top10batches.shape:",neuro_combat_10.shape)
print("===================================================================")
# Data_sub1 = Data_sub.drop(columns=["age", "batch", "sex", "EstimatedTotalIntraCranialVol"])
# feature_name = Data_sub1.columns
# G = len(feature_name)

# print(f"Number of features: {G}")

# print("estimate rmse between neuro-combat and d-combat between two estimations")
# # def RMSE(v1, v2):
# #     return np.sqrt(((v1 - v2) ** 2).mean())
# print("estimate rmse between neuro-combat and d-combat for two sets")
# rmse_10=[]
# rmse_80=[]
# diff=[]
# for g in range(G):
#     v1 = neuro_combat_10.iloc[:, g].to_numpy()
#     v2 = d_combat_10.iloc[:, g].to_numpy()
#     diff1 = np.abs(v1 - v2)
#     max_value1 = np.max(diff1)
#     max_index1 = np.argmax(diff1)
    
#     v3 = neuro_combat_80.iloc[:, g].to_numpy()
#     v4 = d_combat_80.iloc[:, g].to_numpy()
#     diff2 = np.abs(v3 - v4)
#     max_value2 = np.max(diff2)
#     max_index2 = np.argmax(diff2)
    
#     print(f"Column {g}:")
#     print("Maximum abs diff between neuro and d 10:", max_value1)
#     print("Value at max diff location (neuro_combat_10):", v1[max_index1])
#     print("Value at max diff location (d_combat_10):", v2[max_index1])
    
#     print("Maximum abs diff between neuro and d 80:", max_value2)
#     print("Value at max diff location (neuro_combat_80):", v3[max_index2])
#     print("Value at max diff location (d_combat_80):", v4[max_index2])

# print("if there is a difference between neuro_combate 10 anf neuro_combat 80 and simialrly for d-combat")
# diff = pd.concat(diff, axis=0)
# print(d1)
# positive_counts = (diff > 0).sum()
# negative_counts = (diff < 0).sum()
# print(positive_counts, negative_counts)

# print("plot of neuro-combat from 0 and 80 for the first column")
# plt.scatter(neuro_combat_10.iloc[:,0].to_numpy(),neuro_combat_80.iloc[:,0].to_numpy(),s=10)
# plt.plot(range(60000), range(60000), 'r', linestyle='--', label='y=x')
# plt.xlabel("neuro_10")
# plt.ylabel("neuro_80")
# plt.show()



# import itertools
# import matplotlib.cm as cm
# print("plot non-harmonized and harmonized data")

# # Create figure with subplots
# fig, axes = plt.subplots(G, 3, figsize=(20, 6 * G))  # 3 columns: non-harmonized, combat, d-combat
# if G == 1:
#     axes = np.array([axes])  # Ensure consistent indexing

# # Define color mapping
# data = Data_sub
# unique_combinations = [(b, s) for b in data['batch'].unique() for s in np.unique(data['sex'])]
# print(len(unique_combinations))
# color_map = plt.get_cmap('tab20', len(unique_combinations)) 
# color_dict = {combo: color_map(i) for i, combo in enumerate(unique_combinations)}

# legend_entries = {}

# # Iterate over features
# for i in range(G):  
#     x_min, x_max = float('inf'), float('-inf')
#     y_min, y_max = float('inf'), float('-inf')

#     # Find global min/max values for consistent axis scaling
#     for batch in data['batch'].unique():
#         s = np.where(data['batch'] == batch)[0]
#         d = data.iloc[s]
#         age = d['age']
#         y = d[feature_name[i]]
#         y_n = neuro_combat.iloc[s, i]
#         y_d = d_combat.iloc[s, i]
#         x_min = min(x_min, age.min()) - 5
#         x_max = max(x_max, age.max()) + 5
#         y_min = min(y_min, y.min(), y_n.min(), y_d.min()) - 2
#         y_max = max(y_max, y.max(), y_n.max(), y_d.max()) + 2

#     # Second loop for plotting
#     for batch in data['batch'].unique():
#         s = np.where(data['batch'] == batch)[0]
#         d = data.iloc[s]
#         age = d['age']
#         y = d[feature_name[i]]
#         current_sex = d['sex'].values

#         y_n = neuro_combat.iloc[s, i]
#         y_d = d_combat.iloc[s, i]

#         unique_sexes = np.unique(current_sex)

#         for s_value in unique_sexes:
#             indices = np.where(current_sex == s_value)[0]
#             color = color_dict[(batch, s_value)]

#             row = i

#             # Non-harmonized plot
#             ax = axes[row, 0]
#             scatter = ax.scatter(age.iloc[indices], y.iloc[indices], s=30, color=color, label=f'Batch {batch}, Sex {s_value}')
#             ax.set_title(f'Non-Harmonized - {feature_name[i]}')
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             if f'Batch {batch}, Sex {s_value}' not in legend_entries:
#                 legend_entries[f'Batch {batch}, Sex {s_value}'] = scatter

#             # Neuro-Combat plot
#             ax = axes[row, 1]
#             ax.scatter(age.iloc[indices], y_n.iloc[indices], s=30, color=color)
#             ax.set_title(f'Neuro-Combat - {feature_name[i]}')
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             # D-Combat plot
#             ax = axes[row, 2]
#             ax.scatter(age.iloc[indices], y_d.iloc[indices], s=30, color=color)
#             ax.set_title(f'D-Combat - {feature_name[i]}')
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             ax.set_xlabel("Age")
#             ax.set_ylabel(feature_name[i])

# fig.legend(handles=legend_entries.values(), labels=legend_entries.keys(),
#            loc='upper left', bbox_to_anchor=(0.85, 1))

# plt.tight_layout(rect=[0, 0, 0.85, 1])  
# plt.savefig(os.path.join(sites_path, "model_comparison.png"))
# plt.show()