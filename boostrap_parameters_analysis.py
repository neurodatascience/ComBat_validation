#analyze parameters alpha, beta, gamma, and delta.
"""
n_combat_output is a dictionary where each key corresponds to a bootstrap iteration 
(e.g., the first bootstrap, second bootstrap, etc.). 
Each value under a key is itself a dictionary with the following entries:
  - 'combat_data': the harmonized data.
  - 'alpha': the standardized mean.
  - 'beta': the coefficients for the fixed effects ('sex' and 'age').
  - 'XB': the product of the fixed effects and their corresponding coefficients.
  - 'delta_star': the estimated delta parameter.
  - 'gamma_star': the estimated gamma parameter.


d_combat_output contains the d-combat model results.
d_combat_output is a dictionary where each key corresponds to a bootstrap iteration 
(e.g., the first bootstrap, second bootstrap, etc.). 
Each value under a key is itself a dictionary with a key name showing batch ID,
with the following entries:
  - 'combat_data': the harmonized data.
  - 'alpha': the standardized mean.
  - 'beta': the coefficients for the fixed effects ('sex' and 'age').
  - 'XB': the product of the fixed effects and their corresponding coefficients.
  - 'delta_star': the estimated delta parameter.
  - 'gamma_star': the estimated gamma parameter.

"""

import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from bootstrap_helper import neuro_combat_train,d_combat_train

print("===================================================================================")

common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"
ppmi_case_folder_path=os.path.join(common_path,"d-Combat_project","ppmi-age-sex-case-aseg")
#*****************************************************************************************#
save_path=os.path.join(ppmi_case_folder_path,'bootstrap_plot',"44batches")
os.makedirs(save_path,exist_ok=True)
#***************************************************************************************#

print("import ppmi case data")
data_80batches=pd.read_csv(os.path.join(ppmi_case_folder_path,"data_80batches.csv"))
data_80batches=data_80batches.drop(columns=["EstimatedTotalIntraCranialVol"])
feature_name = [col for col in data_80batches.columns if col not in ["batch", "age", "sex"]]

#IDs with group size within each ID
group=data_80batches.groupby('batch').size().reset_index(name="group size")
p_size=np.unique(sorted(group['group size']))

selected_IDs=group[group['group size']>5]#at least 6 participants to ensure model working
print('minimum group size:',selected_IDs["group size"].min())

#data fit to the model
data=data_80batches[data_80batches['batch'].isin(selected_IDs["batch"])]
print("number of batches:",len(data['batch'].unique()))
print("========================================================================================")
file_path=os.path.join(ppmi_case_folder_path,'d_combat_bootstrap_output')
d_output=d_combat_train(data,file_path)
n_output=neuro_combat_train(data)

#**************************************************************#
# ============================================================
# ALPHA: Latent factors (after removing duplicate components)
# ============================================================
# Shape before transpose: (latent_dim × subjects)
# After np.unique(axis=1), duplicate columns are removed

n_alpha = pd.DataFrame(np.unique(n_output['alpha'], axis=1)).reset_index(drop=True)

# d_output has multiple batches; we pick one (e.g., batch 1) since alpha is same across batches
d_alpha = pd.DataFrame(np.unique(d_output[1]['alpha'], axis=1)).reset_index(drop=True)

# ============================================================
# BETA: Covariate effects (same across batches)
# ============================================================
# First row = sex effect, second row = age effect
# Shape: (n_covariates × n_features)

n_beta = pd.DataFrame(n_output['beta'])
n_beta_sex = pd.DataFrame(n_beta.iloc[0, :].to_numpy())
n_beta_age = pd.DataFrame(n_beta.iloc[1, :].to_numpy())

d_beta = pd.DataFrame(d_output[1]['beta'])  # beta is same across batches
d_beta_sex = pd.DataFrame(d_beta.iloc[0, :].to_numpy())
d_beta_age = pd.DataFrame(d_beta.iloc[1, :].to_numpy())

# ============================================================
# GAMMA_STAR: Additive batch effects
# ============================================================
# Shape: (n_batches × n_features)

d_gamma = pd.DataFrame([d_output[k]['gamma_star'] for k in d_output.keys()]).reset_index(drop=True)
n_gamma = pd.DataFrame(n_output['gamma_star']).reset_index(drop=True)

# ============================================================
# DELTA_STAR: Multiplicative batch effects
# ============================================================
# Shape: (n_batches × n_features)

d_delta = pd.DataFrame([d_output[k]['delta_star'] for k in d_output.keys()]).reset_index(drop=True)
n_delta = pd.DataFrame(n_output['delta_star']).reset_index(drop=True)

print("======================================================================================")
"""***"""
N=100#number of times do bootstrap
"""===="""
bootstrap_data_dir=os.path.join(ppmi_case_folder_path,'bootstrap_output',f"{N}_bootstrap_data.pkl")

n_combat_bootstrap_ouput_dir=os.path.join(ppmi_case_folder_path,'n_combat_bootstrap_output',f"{N}_bootstrap_output.pkl")

d_combat_bootstrap_ouput_dir=os.path.join(ppmi_case_folder_path,'d_combat_bootstrap_output',f"{N}_bootstrap_output.pkl")
print("================================================================================================")
#load data

with open(bootstrap_data_dir,'rb') as f:
    bootstrap_data=pickle.load(f)

with open(n_combat_bootstrap_ouput_dir,'rb') as f:
    n_combat_output=pickle.load(f)

with open(d_combat_bootstrap_ouput_dir,'rb') as f:
    d_combat_output=pickle.load(f)  #dict with names indicating the first, second,..,bootstrap 
print("=============================================================================================")
# Alpha: remove duplicated columns before averaging across bootstraps
alpha_n = []
alpha_d = []

for i in range(len(n_combat_output)):  # Loop through bootstraps
    # Remove duplicated columns (latent factors can repeat due to convergence issues)
    unique_alpha_n = np.unique(n_combat_output[i]['alpha'], axis=1)
    unique_alpha_d = np.unique(d_combat_output[i][1]['alpha'], axis=1)

    alpha_n.append(pd.DataFrame(unique_alpha_n))
    alpha_d.append(pd.DataFrame(unique_alpha_d))

# Combine all bootstraps (shape: N_bootstraps × latent_features)
alpha_n = pd.concat(alpha_n, axis=1).T
alpha_d = pd.concat(alpha_d, axis=1).T

# Compute average alpha across bootstraps
alpha_n_avg = pd.DataFrame(alpha_n.mean(axis=0)).reset_index(drop=True)
alpha_d_avg = pd.DataFrame(alpha_d.mean(axis=0)).reset_index(drop=True)
#**********************************************************************************************#
#beta
#**********************************************************************************************#
#sex
beta_n_s = []
beta_d_s = []

for i in range(len(n_combat_output)):  # Loop through bootstrap iterations
    beta_n = n_combat_output[i]['beta'][0, :]# Extract first row of beta (sex effect)
    beta_d = d_combat_output[i][1]['beta'][0, :]
    beta_n_s.append(pd.Series(beta_n))
    beta_d_s.append(pd.Series(beta_d))

# Stack all beta vectors from bootstraps into dataframes
beta_n_s = pd.DataFrame(beta_n_s)  # Shape: N_bootstraps x N_features
beta_d_s = pd.DataFrame(beta_d_s)

# Compute average beta across bootstraps
beta_n_s_avg =pd.DataFrame(beta_n_s.mean(axis=0).to_numpy())
beta_d_s_avg = pd.DataFrame(beta_d_s.mean(axis=0).to_numpy())
#*******************************************************************************#
# Age
beta_n_a = []
beta_d_a = []

for i in range(len(n_combat_output)):  # Loop through bootstraps
    beta_n = n_combat_output[i]['beta'][1, :]  #  age effect
    beta_d = d_combat_output[i][1]['beta'][1, :]  # Batch 1

    beta_n_a.append(pd.Series(beta_n))
    beta_d_a.append(pd.Series(beta_d))

# Stack bootstraps into DataFrames (n_bootstraps × n_features)
beta_n_a = pd.DataFrame(beta_n_a)
beta_d_a = pd.DataFrame(beta_d_a)

# Compute average age beta across bootstraps
beta_n_a_avg = pd.DataFrame(beta_n_a.mean(axis=0).to_numpy())
beta_d_a_avg = pd.DataFrame(beta_d_a.mean(axis=0).to_numpy())
print("======================================================================================")
#gamma
gamma_d={}
for n in range(N):#for bootstraps
    gamma_d[n]={k:d_combat_output[n][k]['gamma_star'] for k in d_combat_output[n].keys()}
    gamma_d[n]=pd.DataFrame(gamma_d[n]).T

gamma_n = {i: pd.DataFrame(n_combat_output[i]['gamma_star']) for i in range(N)}  
#gamma is unque for each feature for each batches

#estimate the accumulated difference between n_gamma, d_gamma and average of gamma_n, gamma_d.
gamma_d1={}
gamma_d_avg={}#avg for each batch each feature
for b in range(n_gamma.shape[0]): # for each batch
    gamma_d1[b]={n:gamma_d[n].iloc[b,:] for n in range(N)}
    gamma_d1[b]=pd.DataFrame(gamma_d1[b]).T#10 x 16

    gamma_d_avg[b]=gamma_d1[b].mean(axis=0)

gamma_d_avg=pd.DataFrame(gamma_d_avg).T #batches x features

gamma_n1={}
gamma_n_avg={}#avg for each batch each feature

for b in range(n_gamma.shape[0]): # for each batch
    gamma_n1[b]={n:gamma_n[n].iloc[b,:] for n in range(N)}
    gamma_n1[b]=pd.DataFrame(gamma_n1[b]).T#10 x 16

    gamma_n_avg[b]=gamma_n1[b].mean(axis=0)

gamma_n_avg=pd.DataFrame(gamma_n_avg).T #batches x features
print("=============================================================")
#delta

delta_d={}
for n in range(N):#for bootstraps
    delta_d[n]={k:d_combat_output[n][k]['delta_star'] for k in d_combat_output[n].keys()}
    delta_d[n]=pd.DataFrame(delta_d[n]).T

delta_n = {i: pd.DataFrame(n_combat_output[i]['delta_star']) for i in range(N)}  
#delta is unque for each feature for each batches
#estimate the accumulated difference between n_delta, d_delta and average of delta_n, delta_d.
delta_d1={}
delta_d_avg={}#avg for each batch each feature

for b in range(n_delta.shape[0]): # for each batch
    delta_d1[b]={n:delta_d[n].iloc[b,:] for n in range(N)}
    delta_d1[b]=pd.DataFrame(delta_d1[b]).T#10 x 16

    delta_d_avg[b]=delta_d1[b].mean(axis=0)

delta_d_avg=pd.DataFrame(delta_d_avg).T #batches x features

delta_n1={}
delta_n_avg={}#avg for each batch each feature

for b in range(n_delta.shape[0]): # for each batch
    delta_n1[b]={n:delta_n[n].iloc[b,:] for n in range(N)}
    delta_n1[b]=pd.DataFrame(delta_n1[b]).T#10 x 16

    delta_n_avg[b]=delta_n1[b].mean(axis=0)

delta_n_avg=pd.DataFrame(delta_n_avg).T #batches x features

#summarize all values into tables and check the influence of the number of batches included in the model
print("Compare alpha, beta, gamma and delta")
print("accumlated difference of alpha")
print("n-combat")
print(n_alpha-alpha_n_avg)
print("d-combat")
print(d_alpha-alpha_d_avg)
print("accumlated difference of beta")
print("sex:n-combat")
print(n_beta_sex-beta_n_s_avg)#for 16 features
print("sex d-combat")
print(d_beta_sex-beta_d_s_avg)
print("age:n-combat")
print(n_beta_age-beta_n_a_avg)
print("age d-combat")
print(d_beta_age-beta_d_a_avg)

print("accumlated difference of gamma")
#difference between n_gamma and gamma_n_avg
diff=(n_gamma-gamma_n_avg).abs().sum(axis=0)#for 16 features
print(f"The accumulated difference between gamma from original data and resampled data from {N} bootstraps across all batches using neuro-combat is:")
print(diff)

#difference between d_gamma and gamma_d_avg
diff1=(d_gamma-gamma_d_avg).abs().sum(axis=0)#for 16 features
print(f"The accumulated difference between gamma from original data and resampled data from {N} bootstraps across all batches using d-combat is:")
print(diff1)
print("accumlated difference of delta")
print("n-combat")
print((n_delta-delta_n_avg).abs().sum(axis=0))
print("d-combat")
print((d_delta-delta_d_avg).abs().sum(axis=0))

####################################
results = {
    "Alpha (n-combat)": np.array(n_alpha - alpha_n_avg).reshape(-1),
    "Alpha (d-combat)": np.array(d_alpha - alpha_d_avg).reshape(-1),
    "Beta sex (n-combat)": np.array(n_beta_sex - beta_n_s_avg).reshape(-1),
    "Beta sex (d-combat)": np.array(d_beta_sex - beta_d_s_avg).reshape(-1),
    "Beta age (n-combat)": np.array(n_beta_age - beta_n_a_avg).reshape(-1),
    "Beta age (d-combat)": np.array(d_beta_age - beta_d_a_avg).reshape(-1),
    "Gamma (n-combat)": np.array((n_gamma - gamma_n_avg).abs().sum(axis=0)).reshape(-1),
    "Gamma (d-combat)": np.array((d_gamma - gamma_d_avg).abs().sum(axis=0)).reshape(-1),
    "Delta (n-combat)": np.array((n_delta - delta_n_avg).abs().sum(axis=0)).reshape(-1),
    "Delta (d-combat)": np.array((d_delta - delta_d_avg).abs().sum(axis=0)).reshape(-1),
}

# Convert to DataFrame
df = pd.DataFrame(results).T
df.columns = [f"Feature {i}" for i in range(16)]

# Show the final table
print("===== Combined Table of Differences =====")
print(df)
df.to_csv(os.path.join(save_path,"parameter_comparison.csv"))

#for the next step, compare with data with first 10, 20, and 30 batches and see if they have difference.