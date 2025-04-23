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

print("===================================================================================")

common_path="/Users/xiaoqixie/Desktop/Mcgill/Winter_Rotation"
result_path=os.path.join(common_path,"d-Combat_project","Bootstrap")
#*****************************************************************************************#
file_name="Homogeneous_Heterogenity_Heterogenity_nonlinear_N140_G3_I5_Gamma4"
save_path=os.path.join(result_path,'bootstrap_plot',file_name)
os.makedirs(save_path,exist_ok=True)
#***************************************************************************************#
data=pd.read_csv(os.path.join(common_path,
                f"combat_sites/min_points28/{file_name}/{file_name}.csv"))
feature_cols=[col for col in data.columns if "feature" in col]
data=data[['batch','sex','age']+feature_cols]
#********************************************************************************************#
alpha=pd.read_csv(os.path.join(common_path,
                f"combat_sites/min_points28/{file_name}/alpha_G.csv")).reset_index(drop=True)

fixed_effects=pd.read_csv(os.path.join(common_path,
                f"combat_sites/min_points28/{file_name}/fixed_effects.csv"))#3*140
# Reset row index
fixed_effects.reset_index(drop=True, inplace=True)

# Reset column index (i.e., rename columns to default integers)
fixed_effects.columns = range(fixed_effects.shape[1])

# print(fixed_effects.shape)
gamma=pd.read_csv(os.path.join(common_path,
                f"combat_sites/min_points28/{file_name}/gamma_IG.csv"))

delta=pd.read_csv(os.path.join(common_path,
                f"combat_sites/min_points28/{file_name}/delta_IG.csv"))

print("========================================================================================")
file_path=os.path.join(result_path,'d_combat_bootstrap_output',file_name)
with open(os.path.join(file_path,"d_output.pkl"),"rb") as f:
    d_output=pickle.load(f)

with open(os.path.join(result_path,"n_combat_bootstrap_output","n_output.pkl"),"rb") as f:
    n_output=pickle.load(f)

keys=list(d_output.keys())
# print(keys)
#**************************************************************#
# ============================================================
# ALPHA: standard mean
# ============================================================
# Shape before transpose: (features × subjects)
# After np.unique(axis=1), duplicate columns are removed

n_alpha = pd.DataFrame(np.unique(n_output['alpha'], axis=1)).reset_index(drop=True)
n_alpha.columns=alpha.columns

# d_output has multiple batches; we pick one (e.g., batch 1) since alpha is same across batches
d_alpha = pd.DataFrame(np.unique(d_output[keys[0]]['alpha'], axis=1)).reset_index(drop=True)
d_alpha.columns=alpha.columns
# ============================================================
# XB: Covariate effects 
# ============================================================
n_XB = pd.DataFrame(n_output['XB']).reset_index(drop=True)#3x140
n_XB.columns=fixed_effects.columns
# print(n_XB.shape)
d_XB = {} 
for k in keys:
    d_XB[k]=pd.DataFrame(d_output[k]['XB'])

d_XB=pd.concat(d_XB,axis=1).reset_index(drop=True)#3x140
d_XB.columns=fixed_effects.columns
# print(d_XB.shape)
# ============================================================
# GAMMA_STAR: Additive batch effects
# ============================================================
# Shape: (n_batches × n_features)

d_gamma = pd.DataFrame([d_output[k]['gamma_star'] for k in d_output.keys()]).reset_index(drop=True).T
d_gamma.columns=gamma.columns
n_gamma = pd.DataFrame(n_output['gamma_star']).reset_index(drop=True).T
n_gamma.columns=gamma.columns
# ============================================================
# DELTA_STAR: Multiplicative batch effects
# ============================================================
# Shape: (n_batches × n_features)

d_delta = pd.DataFrame([d_output[k]['delta_star'] for k in d_output.keys()]).reset_index(drop=True).T
d_delta.columns=delta.columns
n_delta = pd.DataFrame(n_output['delta_star']).reset_index(drop=True).T
n_delta.columns=delta.columns
print("======================================================================================")
"""***"""
N=1000#number of times do bootstrap
"""===="""
bootstrap_data_dir=os.path.join(result_path,'bootstrap_output',
                                file_name,
                                f"{N}_bootstrap_data.pkl")

n_combat_bootstrap_ouput_dir=os.path.join(result_path,
                                          'n_combat_bootstrap_output',
                                          file_name,
                                          f"{N}_bootstrap_output.pkl")

d_combat_bootstrap_ouput_dir=os.path.join(result_path,
                                          'd_combat_bootstrap_output',
                                          file_name,
                                          f"{N}_bootstrap_output.pkl")
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
    unique_alpha_d = np.unique(d_combat_output[i][keys[0]]['alpha'], axis=1)

    alpha_n.append(pd.DataFrame(unique_alpha_n))
    alpha_d.append(pd.DataFrame(unique_alpha_d))

# Combine all bootstraps (shape: N_bootstraps × latent_features)
alpha_n = pd.concat(alpha_n, axis=1).T#1000*3
alpha_d = pd.concat(alpha_d, axis=1).T

# Compute average alpha across bootstraps
alpha_n_avg = pd.DataFrame(alpha_n.mean(axis=0)).reset_index(drop=True)
alpha_d_avg = pd.DataFrame(alpha_d.mean(axis=0)).reset_index(drop=True)
alpha_n_avg.columns=alpha.columns
alpha_d_avg.columns=alpha.columns

# # Compute variance across bootstraps
# alpha_n_std=pd.DataFrame(alpha_n.std(axis=0)).reset_index(drop=True)
# alpha_d_std=pd.DataFrame(alpha_d.std(axis=0)).reset_index(drop=True)
#**********************************************************************************************#
#XB
#**********************************************************************************************#
XB_n = []
XB_d = []

for i in range(len(n_combat_output)):  # Loop through bootstrap iterations
    xb_n = pd.DataFrame(n_combat_output[i]['XB'])# Extract first row of beta (sex effect)
    xb_d={}
    for k in keys:
        xb_d[k] = d_combat_output[i][k]['XB']
    xb_d=pd.concat(xb_d,axis=1)
    XB_n.append(xb_n)
    XB_d.append(xb_d)
XB_n_avg={}
XB_d_avg={}
for f in range(len(feature_cols)):
    xb_n=[]
    xb_d=[]
    for i in range(len(n_combat_output)):
        xb_n.append(XB_n[i].iloc[f,:])
        xb_d.append(XB_d[i].iloc[f,:])
    xb_n=pd.concat(xb_n,axis=1)#140*1000
    xb_d=pd.concat(xb_d,axis=1)
    XB_n_avg[f]=xb_n.mean(axis=1)
    XB_d_avg[f]=xb_d.mean(axis=1)
XB_n_avg=pd.concat(XB_n_avg,axis=1).T.reset_index(drop=True)#3*140
XB_n_avg.columns=fixed_effects.columns
XB_d_avg=pd.concat(XB_d_avg,axis=1).T.reset_index(drop=True)
XB_d_avg.columns=fixed_effects.columns
#======================================================================================#
#gamma
gamma_d={}
for n in range(N):#for bootstraps
    gamma_d[n]={k:d_combat_output[n][k]['gamma_star'] for k in d_combat_output[n].keys()}
    gamma_d[n]=pd.DataFrame(gamma_d[n]).T

gamma_n = {i: pd.DataFrame(n_combat_output[i]['gamma_star']) for i in range(N)}  
#gamma is unque for each feature for each batches

#estimate the accumulated difference between n_gamma, d_gamma and average of gamma_n, gamma_d.
gamma_d1={}
gamma_d_std={}#avg std for gamma_ig over i
gamma_d_avg={}#avg for each batch each feature
for b in range(n_gamma.shape[1]):
    gamma_d1[b]={n:gamma_d[n].iloc[b,:] for n in range(N)}
    gamma_d1[b]=pd.DataFrame(gamma_d1[b]).T
    gamma_d_std[b]=pd.Series(gamma_d1[b].std(axis=0))#std of features within a batch over N bootstraps

    gamma_d_avg[b]=gamma_d1[b].mean(axis=0)

gamma_d_avg=pd.DataFrame(gamma_d_avg).reset_index(drop=True) 

gamma_d_avg.columns=gamma.columns
gamma_d_std=pd.DataFrame(gamma_d_std).reset_index(drop=True) 

gamma_n1={}
gamma_n_avg={}
gamma_n_std={}

for b in range(n_gamma.shape[1]): 
    gamma_n1[b]={n:gamma_n[n].iloc[b,:] for n in range(N)}
    gamma_n1[b]=pd.DataFrame(gamma_n1[b]).T

    gamma_n_std[b]=pd.Series(gamma_n1[b].std(axis=0))

    gamma_n_avg[b]=gamma_n1[b].mean(axis=0)

gamma_n_avg=pd.DataFrame(gamma_n_avg).reset_index(drop=True) 
gamma_n_avg.columns=gamma.columns
gamma_n_std=pd.DataFrame(gamma_n_std).reset_index(drop=True) 
#=============================================================#
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
delta_d_std={}

for b in range(n_delta.shape[1]): # for each batch
    delta_d1[b]={n:delta_d[n].iloc[b,:] for n in range(N)}
    delta_d1[b]=pd.DataFrame(delta_d1[b]).T#10 x 16

    delta_d_std[b]=pd.Series(delta_d1[b].std(axis=0))

    delta_d_avg[b]=delta_d1[b].mean(axis=0)

delta_d_avg=pd.DataFrame(delta_d_avg).reset_index(drop=True)
delta_d_avg.columns=delta.columns
delta_d_std=pd.DataFrame(delta_d_std).reset_index(drop=True)

delta_n1={}
delta_n_avg={}#avg for each batch each feature
delta_n_std={}

for b in range(n_delta.shape[1]): # for each batch
    delta_n1[b]={n:delta_n[n].iloc[b,:] for n in range(N)}
    delta_n1[b]=pd.DataFrame(delta_n1[b]).T

    delta_n_std[b]=pd.Series(delta_n1[b].std(axis=0))

    delta_n_avg[b]=delta_n1[b].mean(axis=0)

delta_n_avg=pd.DataFrame(delta_n_avg).reset_index(drop=True) 
delta_n_avg.columns=delta.columns
delta_n_std=pd.DataFrame(delta_n_std).reset_index(drop=True) 

#summarize all values into tables and check the influence of the number of batches included in the model
print("Compare alpha, beta, gamma and delta")
print("accumlated difference of alpha")
print("n-combat")
print(alpha-alpha_n_avg)
print("d-combat")
print(alpha-alpha_d_avg)
print("accumlated difference of XB")
print("XB:n-combat")
print((fixed_effects-XB_n_avg).abs().mean(axis=1))
print("XB d-combat")
print((fixed_effects-XB_d_avg).abs().mean(axis=1))
print("diff between fixed efects and d_XB")
print((fixed_effects-d_XB).abs().mean(axis=1))
print("accumlated difference of gamma")


print("diff between gamma and gamma_n_avg")
print((gamma-gamma_n_avg).abs().mean(axis=1))

print("diff between gamma and gamma_d_avg")
print((gamma-gamma_d_avg).abs().mean(axis=1))

print("diff between gamma and gamma_n")
print((gamma-d_gamma).abs().mean(axis=1))

print("diff between gamma and gamma_d")
print((gamma-d_gamma).abs().mean(axis=1))


print("diff between delta and delta_n_avg")
print((delta-delta_n_avg).abs().mean(axis=1))

print("diff between delta and delta_d_avg")
print((delta-delta_d_avg).abs().mean(axis=1))

print("diff between delta and delta_n")
print((delta-d_delta).abs().mean(axis=1))

print("diff between delta and delta_d")
print((delta-d_delta).abs().mean(axis=1))


# ####################################
# results = {
#     "Alpha (n-combat)":n_alpha.to_numpy().flatten(),
#     "Alpha (n-combat bootstrap avg)": alpha_n_avg.to_numpy().flatten(),
#     "Diff_a_n": n_alpha.to_numpy().flatten()-alpha_n_avg.to_numpy().flatten(),
#     "Alpha (n-combat bootstrap std)": alpha_n_std.to_numpy().flatten(),

#     "Alpha (d-combat)": d_alpha.to_numpy().flatten(),
#     "Alpha (d-combat bootstrap avg)": alpha_d_avg.to_numpy().flatten(),
#     "Diff_a_d":d_alpha.to_numpy().flatten()-alpha_d_avg.to_numpy().flatten(),
#     "Alpha (d-combat bootstrap std)": alpha_d_std.to_numpy().flatten(),

#     "Beta sex (n-combat)": n_beta_sex.to_numpy().flatten(),
#     "Beta sex (n-combat bootstrap avg)": beta_n_s_avg.to_numpy().flatten(),
#     "Diff_b_s_n":n_beta_sex.to_numpy().flatten()-beta_n_s_avg.to_numpy().flatten(),
#     "Beta sex (n-combat bootstrap std)": beta_n_s_std.to_numpy().flatten(),

#     "Beta sex (d-combat)": d_beta_sex.to_numpy().flatten(),
#     "Beta sex (d-combat bootstrap avg)": beta_d_s_avg.to_numpy().flatten(),
#     "Diff_b_s_d": d_beta_sex.to_numpy().flatten()-beta_d_s_avg.to_numpy().flatten(),
#     "Beta sex (d-combat bootstrap std)": beta_d_s_std.to_numpy().flatten(),

#     "Beta age (n-combat)": n_beta_age.to_numpy().flatten(),
#     "Beta age (n-combat bootstrap avg)": beta_n_a_avg.to_numpy().flatten(),
#     "Diff_b_a_n":n_beta_age.to_numpy().flatten()-beta_n_a_avg.to_numpy().flatten(),
#     "Beta age (n-combat bootstrap std)": beta_n_a_std.to_numpy().flatten(),

#     "Beta age (d-combat)": d_beta_age.to_numpy().flatten(),
#     "Beta age (d-combat bootstrap avg)": beta_d_a_avg.to_numpy().flatten(),
#     "Diff_b_a_d": d_beta_age.to_numpy().flatten()-beta_d_a_avg.to_numpy().flatten(),
#     "Beta age (d-combat bootstrap std)": beta_d_a_std.to_numpy().flatten(),

#     "Gamma (n-combat)": pd.DataFrame(n_gamma.mean(axis=0)).to_numpy().flatten(),
#     "Gamma (n-combat bootstrap avg)": pd.DataFrame(gamma_n_avg.mean(axis=0)).to_numpy().flatten(),
#     "Diff_g_n": pd.DataFrame(n_gamma.mean(axis=0)).to_numpy().flatten()-pd.DataFrame(gamma_n_avg.mean(axis=0)).to_numpy().flatten(),
#     "Gamma (n-combat bootstrap std)": pd.DataFrame(gamma_n_std.mean(axis=0)).to_numpy().flatten(),

#     "Gamma (d-combat)": pd.DataFrame(d_gamma.mean(axis=0)).to_numpy().flatten(),
#     "Gamma (d-combat bootstrap avg)": pd.DataFrame(gamma_d_avg.mean(axis=0)).to_numpy().flatten(),
#     "Diff_g_d":pd.DataFrame(d_gamma.mean(axis=0)).to_numpy().flatten()-pd.DataFrame(gamma_d_avg.mean(axis=0)).to_numpy().flatten(),
#     "Gamma (d-combat bootstrap std)": pd.DataFrame(gamma_d_std.mean(axis=0)).to_numpy().flatten(),
    
#     "Delta (n-combat)": pd.DataFrame(n_delta.mean(axis=0)).to_numpy().flatten(),
#     "Delta (n-combat bootstrap avg)": pd.DataFrame(delta_n_avg.mean(axis=0)).to_numpy().flatten(),
#     "Diff_d_n": pd.DataFrame(n_delta.mean(axis=0)).to_numpy().flatten()-pd.DataFrame(delta_n_avg.mean(axis=0)).to_numpy().flatten(),
#     "Delta (n-combat bootstrap std)": pd.DataFrame(delta_n_std.mean(axis=0)).to_numpy().flatten(),

#     "Delta (d-combat)": pd.DataFrame(d_delta.mean(axis=0)).to_numpy().flatten(),
#     "Delta (d-combat bootstrap avg)": pd.DataFrame(delta_d_avg.mean(axis=0)).to_numpy().flatten(),
#     "Diff_d_d":pd.DataFrame(d_delta.mean(axis=0)).to_numpy().flatten()-pd.DataFrame(delta_d_avg.mean(axis=0)).to_numpy().flatten(),
#     "Delta (d-combat bootstrap std)": pd.DataFrame(delta_d_std.mean(axis=0)).to_numpy().flatten(),
# }

# df=pd.DataFrame(results).T
# df.columns=[f"feature{i}" for i in range(len(feature_cols))]
# # Show the final table
# print("===== Combined Table of Differences =====")
# print(df)
# df.to_csv(os.path.join(save_path,"parameter_comparison.csv"))

