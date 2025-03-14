import numpy as np
import pandas as pd
import scipy.stats as stats
import os

from resample_helper import neuro_combat,bootstrap_with_noise,d_combat 

common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"
ppmi_case_folder_path=os.path.join(common_path,"d-Combat_project","ppmi-age-sex-case-aseg")

print("import ppmi case data")
data_80batches=pd.read_csv(os.path.join(ppmi_case_folder_path,"data_80batches.csv"))
data_80batches=data_80batches.drop(columns=["EstimatedTotalIntraCranialVol"])

print("data with at least 40 samples")
group=data_80batches.groupby("batch").size().reset_index(name="sample size")
id_geq40=group[group["sample size"]>=40]["batch"]
# print(id_geq40)
data_geq40=pd.DataFrame(data_80batches[data_80batches["batch"].isin(id_geq40)])
print("data_geq40.shape:",data_geq40.shape)

print("=====================================================================================")
print("estimate parameters from combat model")
# output=neuro_combat(data_geq40)
os.makedirs(os.path.join(ppmi_case_folder_path,"dCombat_sites","original_data"),exist_ok=True)
output=d_combat(data_geq40,os.path.join(ppmi_case_folder_path,"dCombat_sites","original_data"))

def d_combat_param(output,ids):
    #output: d_combat output
    #ids: batch id we used in d_combat model
    sigma=output[0]["estimates"]["var_pooled"]#same for different site
    delta={}
    gamma={}
    for i,b in enumerate(ids):
        delta[b]=output[i]["estimates"]["delta_star"]
        gamma[b]=output[i]["estimates"]["gamma_star"]
        # print(gamma[b])
    delta=pd.DataFrame(delta).T
    gamma=pd.DataFrame(gamma).T
    return sigma, delta, gamma
sigma, delta, gamma=d_combat_param(output,id_geq40)
print("============================================================================================================")
print("add noise follows N(0,sigma_hat) for each feature")

feature_cols = [col for col in data_geq40.columns if col not in ["batch", "age", "sex"]]
# bootstrap for each batch for each sex
sex=data_geq40["sex"].unique()
n=30#30 samples per sex
ntimes=1000
version=1
print('=======================================================================')
print("esitmate variance within groups that make by considering unique combination of feature, batch and sex.")

var_type="not_feature"

if var_type=="not_feature":
    sigma_for_b = {}
    for ID in id_geq40:
        var = {}  
        
        for s in sex:
            idx = np.where((data_geq40["batch"] == ID) & (data_geq40["sex"] == s))[0] 
            data_sub = data_geq40.iloc[idx, :][feature_cols]  
            
            if len(data_sub) > 1:  
                var[s] = data_sub.var(axis=0)#variance of each feature for each sex for each feature
                print(f"Batch {ID}, Sex {s}: Variance {var[s].shape}")
            else:
                print(f"Batch {ID}, Sex {s}: Not enough samples to compute variance.")

        sigma_for_b[ID] = var 
if var_type=="feature":
    sigma_for_b=sigma

"""possibly the issue of lowered variance is from repeated samples"""
# print("==========================================================================================")
# print('do 1000 times bootstrap')#compare with gamma hat
# bootstrap_data={}
# b_output={}
# model_type="d_combat"
# np.random.seed(123)
# for i in range(ntimes):
#     bootstrap_data[i]=bootstrap_with_noise(data_geq40,n,sigma_for_b,var_type)
#     print("shape of bootstrap data:",bootstrap_data[i].shape)
#     if model_type=="n_combat":
#         b_output[i]=neuro_combat(bootstrap_data[i])
#     if model_type=="d_combat":
#         folder_path=os.path.join(ppmi_case_folder_path,"dCombat_sites","resampled_data",f"test{version}",f"set_{i}")
#         os.makedirs(folder_path,exist_ok=True)
#         b_output[i]=d_combat(bootstrap_data[i],folder_path)
# print("========================================================================================================")
# b_gamma={}
# b_delta={}
# b_sigma={}
# for i in range(len(b_output)):
#     if model_type=="n_combat":
#         b_gamma[i]=b_output[i]["estimates"]["gamma.star"]
#         b_delta[i]=b_output[i]["estimates"]["delta.star"]
#         b_sigma[i]=b_output[i]["estimates"]["var_pooled"]
#     if model_type=="d_combat":
#         b_sigma[i], b_delta[i], b_gamma[i]=d_combat_param(b_output[i],id_geq40)
#         print(b_delta[i])

# b_gamma_array = np.array(list(b_gamma.values()))
# b_gamma_mean = np.mean(b_gamma_array, axis=0) 

# b_delta_array = np.array(list(b_delta.values()))
# b_delta_mean = np.mean(b_delta_array, axis=0) 

# print("b_gamma_mean.shape:",b_gamma_mean.shape)
# print("b_delta_mean.shape:",b_delta_mean.shape)
# print("==========================================================================")
# print("compare gamma from output and d-output")
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(len(feature_cols), 1, figsize=(8, len(feature_cols) * 4), sharex=False)

# for g, feature in enumerate(feature_cols):
#     ax = axes[g]  # Select subplot
#     ax.scatter(id_geq40.astype(str), gamma.iloc[:, g], color="black", label="Original data gamma")
#     ax.scatter(id_geq40.astype(str), b_gamma_mean[:, g], color="red", label="Resampled data gamma",alpha=0.5,s=40)
#     ax.set_ylabel('gamma')
#     ax.set_title(f'Feature: {feature}')
#     ax.legend()

# axes[-1].set_xlabel('Batch ID')  # Set xlabel on the last subplot
# plt.tight_layout()
# save_path=os.path.join(ppmi_case_folder_path,"gamma_comparisons_original_resample.png")
# plt.savefig(save_path)
# plt.show()

# print("diff between gamma and b_gamma1 and gamma and b_gamma2")
# print("1",np.mean(np.abs(gamma-b_gamma_mean)),np.var(np.abs(gamma-b_gamma_mean)))

# print("how about delta")
# print("compare delta from output and d-output")
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(len(feature_cols), 1, figsize=(8, len(feature_cols) * 4), sharex=False)

# for g, feature in enumerate(feature_cols):
#     ax = axes[g]  # Select subplot
#     ax.scatter(id_geq40.astype(str), delta.iloc[:, g], color="black", label="Original data delta")
#     ax.scatter(id_geq40.astype(str), b_delta_mean[:, g], color="red", label="Resampled data delta",alpha=0.5,s=40)
#     ax.set_ylabel('delta')
#     ax.set_title(f'Feature: {feature}')
#     ax.legend()

# axes[-1].set_xlabel('Batch ID')  # Set xlabel on the last subplot
# plt.tight_layout()
# save_path=os.path.join(ppmi_case_folder_path,"delta_comparisons_original_resample.png")
# plt.savefig(save_path)
# plt.show()


