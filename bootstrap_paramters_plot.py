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
N=100#number of times do bootstrap

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

print("==================================================================================")
#alpha 
alpha_n=[]
alpha_d=[]
for i in range(len(n_combat_output)):#bootstraps
    alpha_n.append(pd.DataFrame((np.unique(n_combat_output[i]['alpha'],axis=1))))#row avg
    alpha_d.append(pd.DataFrame((np.unique(d_combat_output[i][1]['alpha'],axis=1))))#alpha sotred in different batches are same

alpha_n=pd.concat(alpha_n,axis=1).T#N x features
alpha_n_avg=pd.Series(alpha_n.mean(axis=0)).reset_index(drop=True)#features
alpha_n.columns = [f'feature{i}' for i in range(alpha_n.shape[1])]

alpha_d=pd.concat(alpha_d,axis=1).T
alpha_d_avg=pd.Series(alpha_d.mean(axis=0)).reset_index(drop=True)#features
alpha_d.columns = [f'feature{i}' for i in range(alpha_d.shape[1])]

x_label = [f'feature{i}' for i in range(alpha_n.shape[1])]
# ========== Confidence bounds ==========
p1, p2 = 0.1, 0.9
bounds_n = {
    f: np.quantile(alpha_n.iloc[:, f], [p1, p2])
    for f in range(alpha_n.shape[1])
}
bounds_d = {
    f: np.quantile(alpha_d.iloc[:, f], [p1, p2])
    for f in range(alpha_d.shape[1])
}

bounds_n = pd.DataFrame(bounds_n).reset_index(drop=True)
bounds_d = pd.DataFrame(bounds_d).reset_index(drop=True)

# print(bounds_n)
#******************************************************************#
#neuro plot
# ========== Prepare for plotting ==========
# Compute asymmetric error bars
yerr_lower1 = pd.Series(alpha_n_avg - bounds_n.iloc[0])
yerr_upper1 = pd.Series(bounds_n.iloc[1]- alpha_n_avg)

num_features = len(feature_name)

feature_name_f=pd.Series(feature_name)

n_cols = 4
n_rows = int(np.ceil(num_features / n_cols))

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=False)
axs = axs.flatten()  # Flatten the 2D array of axes

if num_features == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    feature_label = feature_name_f[i]
    
    # Plot error bar for bootstrapped mean
    ax.errorbar(0, alpha_n_avg.iloc[i],
                yerr=[[yerr_lower1.iloc[i]], [yerr_upper1.iloc[i]]],
                fmt='o', color='red', ecolor='black', elinewidth=1.5, capsize=4,
                label=f'Mean ± 10th–90th percentile\nfrom {N} bootstrap')

    # Plot original data alpha
    ax.scatter(0, n_alpha.iloc[i], color='green', label='Original data alpha',marker='x',s=50)

    # Customize subplot
    ax.set_ylabel('Alpha value')
    ax.set_title(f'Feature: {feature_label}')
    ax.set_xticks([0])
    ax.set_xticklabels([feature_label])
    ax.grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.85, 1), frameon=False)
plt.tight_layout()
plt.subplots_adjust(right=0.85)  
plt.savefig(os.path.join(save_path,f'b{N}_alpha_n.png'), bbox_inches='tight')
# plt.show()
#******************************************************************#
#d-combat plot
yerr_lower2 = pd.Series(alpha_d_avg - bounds_d.iloc[0])
yerr_upper2 = pd.Series(bounds_d.iloc[1] - alpha_d_avg)

num_features = len(feature_name)

n_cols = 4
n_rows = int(np.ceil(num_features / n_cols))

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=False)
axs = axs.flatten()  # Flatten the 2D array of axes

if num_features == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    feature_label = feature_name_f[i]
    
    # Plot error bar for bootstrapped mean
    ax.errorbar(0, alpha_d_avg.iloc[i],
                yerr=[[yerr_lower2.iloc[i]], [yerr_upper2.iloc[i]]],
                fmt='o', color='red', ecolor='black', elinewidth=1.5, capsize=4,
                label=f'Mean ± 10th–90th percentile\nfrom {N} bootstrap')

    # Plot original data alpha
    ax.scatter(0, d_alpha.iloc[i], color='green', label='Original data alpha',marker='x',s=50)

    # Customize subplot
    ax.set_ylabel('Alpha value')
    ax.set_title(f'Feature: {feature_label}')
    ax.set_xticks([0])
    ax.set_xticklabels([feature_label])
    ax.grid(True)
    

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.85, 1), frameon=False)
plt.tight_layout()
plt.subplots_adjust(right=0.85)  
plt.savefig(os.path.join(save_path,f'b{N}_alpha_d.png'), bbox_inches='tight')
# plt.show()
print("=====================================================================================")
#beta
#**********************************************************************************************#
#sex
beta_n=[]
beta_d=[]
for i in range(len(n_combat_output)):#bootstraps
    beta_n.append(pd.DataFrame(n_combat_output[i]['beta'][0,:]))#row avg
    beta_d.append(pd.DataFrame(d_combat_output[i][1]['beta'][0,:]))#beta sotred in different batches are same

beta_n=pd.concat(beta_n,axis=1).T#N x features
beta_n_avg=pd.Series(beta_n.mean(axis=0).to_numpy())#features
beta_n.columns=[f'feature{i}' for i in range(len(beta_n_avg))]

beta_d=pd.concat(beta_d,axis=1).T
beta_d_avg=pd.Series(beta_d.mean(axis=0).to_numpy())#features
beta_d.columns=[f'feature{i}' for i in range(len(beta_d_avg))]

x_label=[f'feature{i}' for i in range(len(beta_n))]

p1=0.1
p2=0.9

bounds_n={}
bounds_d={}
for f in range(len(beta_n_avg)):
    bounds_n[f]=np.quantile(beta_n.iloc[:,f],[p1,p2])
    bounds_d[f]=np.quantile(beta_d.iloc[:,f],[p1,p2])

bounds_n=pd.DataFrame(bounds_n).T.reset_index(drop=True)
bounds_d=pd.DataFrame(bounds_d).T.reset_index(drop=True)

#neuro plot
lower_bounds_filtered = np.array(bounds_n.iloc[:,0])
upper_bounds_filtered = np.array(bounds_n.iloc[:,1])

yerr_lower = beta_n_avg - lower_bounds_filtered
yerr_upper = upper_bounds_filtered - beta_n_avg

num_features = len(feature_name)

n_cols = 4
n_rows = int(np.ceil(num_features / n_cols))

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=False)
axs = axs.flatten()  # Flatten the 2D array of axes

if num_features == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    feature_label = feature_name_f[i]
    
    # Plot error bar for bootstrapped mean
    ax.errorbar(0, beta_n_avg.iloc[i],
                yerr=[[yerr_lower.iloc[i]], [yerr_upper.iloc[i]]],
                fmt='o', color='red', ecolor='black', elinewidth=1.5, capsize=4,
                label=f'Mean ± 10th–90th percentile\nfrom {N} bootstrap')

    # Plot original data beta
    ax.scatter(0, n_beta_sex.iloc[i], color='green', label='Original data beta',marker='x',s=50)

    # Customize subplot
    ax.set_ylabel('beta value')
    ax.set_title(f'Feature: sex-{feature_label}')
    ax.set_xticks([0])
    ax.set_xticklabels([feature_label])
    ax.grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.85, 1), frameon=False)
plt.tight_layout()
plt.subplots_adjust(right=0.85)  
plt.savefig(os.path.join(save_path,f'b{N}_beta_sex_n.png'), bbox_inches='tight')

#d plot
lower_bounds_filtered = bounds_n.iloc[:,0]
upper_bounds_filtered = bounds_n.iloc[:,1]


yerr_lower = beta_d_avg - lower_bounds_filtered
yerr_upper = upper_bounds_filtered - beta_d_avg
asymmetric_error = [yerr_lower, yerr_upper]

n_cols = 4
n_rows = int(np.ceil(num_features / n_cols))

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=False)
axs = axs.flatten()  # Flatten the 2D array of axes

if num_features == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    feature_label = feature_name_f[i]
    
    # Plot error bar for bootstrapped mean
    ax.errorbar(0, beta_d_avg.iloc[i],
                yerr=[[yerr_lower.iloc[i]], [yerr_upper.iloc[i]]],
                fmt='o', color='red', ecolor='black', elinewidth=1.5, capsize=4,
                label=f'Mean ± 10th–90th percentile\nfrom {N} bootstrap')

    # Plot original data beta
    ax.scatter(0, d_beta_sex.iloc[i], color='green', label='Original data beta',marker='x',s=50)

    # Customize subplot
    ax.set_ylabel('beta value')
    ax.set_title(f'Feature: sex-{feature_label}')
    ax.set_xticks([0])
    ax.set_xticklabels([feature_label])
    ax.grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.85, 1), frameon=False)
plt.tight_layout()
plt.subplots_adjust(right=0.85)  
plt.savefig(os.path.join(save_path,f'b{N}_beta_sex_d.png'), bbox_inches='tight')
#**********************************************************************************************#
#age
beta_n=[]
beta_d=[]
for i in range(len(n_combat_output)):#bootstraps
    beta_n.append(pd.DataFrame(n_combat_output[i]['beta'][1,:]))#row avg
    beta_d.append(pd.DataFrame(d_combat_output[i][1]['beta'][1,:]))#beta sotred in different batches are same

beta_n=pd.concat(beta_n,axis=1).T#N x features
beta_n_avg=pd.Series(beta_n.mean(axis=0).to_numpy())#features
beta_n.columns=[f'feature{i}' for i in range(len(beta_n_avg))]

beta_d=pd.concat(beta_d,axis=1).T
beta_d_avg=pd.Series(beta_d.mean(axis=0).to_numpy())#features
beta_d.columns=[f'feature{i}' for i in range(len(beta_d_avg))]

x_label=[f'feature{i}' for i in range(len(beta_n))]

p1=0.1
p2=0.9

bounds_n={}
bounds_d={}
for f in range(len(beta_n_avg)):
    bounds_n[f]=np.quantile(beta_n.iloc[:,f],[p1,p2])
    bounds_d[f]=np.quantile(beta_d.iloc[:,f],[p1,p2])

bounds_n=pd.DataFrame(bounds_n).T.reset_index(drop=True)
bounds_d=pd.DataFrame(bounds_d).T.reset_index(drop=True)

#neuro plot
lower_bounds= np.array(bounds_n.iloc[:,0])
upper_bounds = np.array(bounds_n.iloc[:,1])

yerr_lower = beta_n_avg - lower_bounds
yerr_upper = upper_bounds - beta_n_avg
asymmetric_error = [yerr_lower, yerr_upper]

n_cols = 4
n_rows = int(np.ceil(num_features / n_cols))

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=False)
axs = axs.flatten()  # Flatten the 2D array of axes

if num_features == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    feature_label = feature_name_f[i]
    
    # Plot error bar for bootstrapped mean
    ax.errorbar(0, beta_n_avg.iloc[i],
                yerr=[[asymmetric_error[0].iloc[i]], [asymmetric_error[1].iloc[i]]],
                fmt='o', color='red', ecolor='black', elinewidth=1.5, capsize=4,
                label=f'Mean ± 10th–90th percentile\nfrom {N} bootstrap')

    # Plot original data beta
    ax.scatter(0, n_beta_age.iloc[i], color='green', label='Original data beta',marker='x',s=50)

    # Customize subplot
    ax.set_ylabel('beta value')
    ax.set_title(f'Feature: age-{feature_label}')
    ax.set_xticks([0])
    ax.set_xticklabels([feature_label])
    ax.grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.85, 1), frameon=False)
plt.tight_layout()
plt.subplots_adjust(right=0.85)  
plt.savefig(os.path.join(save_path,f'b{N}_beta_age_n.png'), bbox_inches='tight')

#d plot
#neuro plot
lower_bounds = np.array(bounds_d.iloc[:,0])
upper_bounds = np.array(bounds_d.iloc[:,1])


yerr_lower = beta_d_avg - lower_bounds
yerr_upper = upper_bounds- beta_d_avg
asymmetric_error = [yerr_lower, yerr_upper]

n_cols = 4
n_rows = int(np.ceil(num_features / n_cols))

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=False)
axs = axs.flatten()  # Flatten the 2D array of axes

if num_features == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    feature_label = feature_name_f[i]
    
    # Plot error bar for bootstrapped mean
    ax.errorbar(0, beta_d_avg.iloc[i],
                yerr=[[asymmetric_error[0].iloc[i]], [asymmetric_error[1].iloc[i]]],
                fmt='o', color='red', ecolor='black', elinewidth=1.5, capsize=4,
                label=f'Mean ± 10th–90th percentile\nfrom {N} bootstrap')

    # Plot original data beta
    ax.scatter(0, d_beta_age.iloc[i], color='green', label='Original data beta',marker='x',s=50)

    # Customize subplot
    ax.set_ylabel('beta value')
    ax.set_title(f'Feature: age-{feature_label}')
    ax.set_xticks([0])
    ax.set_xticklabels([feature_label])
    ax.grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.85, 1), frameon=False)
plt.tight_layout()
plt.subplots_adjust(right=0.85)  
plt.savefig(os.path.join(save_path,f'b{N}_beta_age_d.png'), bbox_inches='tight')
print("======================================================================================")
#gamma
gamma_d={}
for n in range(N):#for bootstraps
    gamma_d[n]={k:d_combat_output[n][k]['gamma_star'] for k in d_combat_output[n].keys()}
    gamma_d[n]=pd.DataFrame(gamma_d[n]).T

gamma_n = {i: pd.DataFrame(n_combat_output[i]['gamma_star']) for i in range(N)}  
#gamma is unque for each feature for each batches

p1=0.1
p2=0.9
#estimate the accumulated difference between n_gamma, d_gamma and average of gamma_n, gamma_d.
gamma_d1={}
gamma_d_avg={}#avg for each batch each feature

gamma_d_ci_low = {}
gamma_d_ci_high = {}

for b in range(n_gamma.shape[0]): # for each batch
    gamma_d1[b]={n:gamma_d[n].iloc[b,:] for n in range(N)}
    gamma_d1[b]=pd.DataFrame(gamma_d1[b]).T#10 x 16

    gamma_d_ci_low[b] = gamma_d1[b].quantile(p1)
    gamma_d_ci_high[b] = gamma_d1[b].quantile(p2)

    gamma_d_avg[b]=gamma_d1[b].mean(axis=0)

gamma_d_avg=pd.DataFrame(gamma_d_avg).T #batches x features

gamma_n1={}
gamma_n_avg={}#avg for each batch each feature

gamma_n_ci_low = {}
gamma_n_ci_high = {}

for b in range(n_gamma.shape[0]): # for each batch
    gamma_n1[b]={n:gamma_n[n].iloc[b,:] for n in range(N)}
    gamma_n1[b]=pd.DataFrame(gamma_n1[b]).T#10 x 16

    gamma_n_ci_low[b] = gamma_n1[b].quantile(p1)
    gamma_n_ci_high[b] = gamma_n1[b].quantile(p2)

    gamma_n_avg[b]=gamma_n1[b].mean(axis=0)

gamma_n_avg=pd.DataFrame(gamma_n_avg).T #batches x features

#=============================================================================#
#assign color for batches
num_features=len(feature_name)
keys = list(d_combat_output[1].keys())
num_keys = len(keys)
cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(i / num_keys) for i in range(num_keys)]

#plot of batches for the same feature in the same plot
fig, axs = plt.subplots(16, 1, figsize=(12, 25), sharex=False)
axs = axs.flatten()

# Convert keys to consistent discrete x values
x_vals = list(range(len(keys))) 
x_labels = keys  # your actual batch IDs

#neuro-combat
for i in range(num_features):  # for each feature
    for b in range(n_gamma.shape[0]):  # for each batch
        mean_val = gamma_n_avg.iloc[b, i]
        low = gamma_n_ci_low[b].iloc[i]
        high = gamma_n_ci_high[b].iloc[i]

        axs[i].errorbar(
            x_vals[b], mean_val,
            yerr=[[mean_val - low], [high - mean_val]],
            fmt='o', color=colors[b], capsize=3,
            label=f'Batch {keys[b]}' if i == 0 else ""
        )

        # Plot original data beta
        axs[i].scatter(x_vals[b], n_gamma.iloc[b,i], color='green',marker='x',s=50)


    axs[i].set_ylabel(feature_name[i])
    axs[i].set_xticks(x_vals)
    axs[i].set_xticklabels(x_labels, rotation=45)
    axs[i].grid(True)

axs[-1].set_xlabel("Batch ID")
axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
fig.subplots_adjust(hspace=0.5) 
plt.savefig(os.path.join(save_path,f"b{N}_gamma_n.png"))
plt.close()


#plot of batches for the same feature in the same plot
fig, axs = plt.subplots(16, 1, figsize=(12, 25), sharex=False)
axs = axs.flatten()

# Convert keys to consistent discrete x values
x_vals = list(range(len(keys))) 
x_labels = keys  # your actual batch IDs

#d-combat
for i in range(num_features):  # for each feature
    for b in range(n_gamma.shape[0]):  # for each batch
        mean_val = gamma_d_avg.iloc[b, i]
        low = gamma_d_ci_low[b].iloc[i]
        high = gamma_d_ci_high[b].iloc[i]

        axs[i].errorbar(
            x_vals[b], mean_val,
            yerr=[[mean_val - low], [high - mean_val]],
            fmt='o', color=colors[b], capsize=3,
            label=f'Batch {keys[b]}' if i == 0 else ""
        )
        # Plot original data beta
        axs[i].scatter(x_vals[b], d_gamma.iloc[b,i], color='green', marker='x',s=50)

    axs[i].set_ylabel(feature_name[i])
    axs[i].set_xticks(x_vals)
    axs[i].set_xticklabels(x_labels, rotation=45)
    axs[i].grid(True)

axs[-1].set_xlabel("Batch ID")
axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
fig.subplots_adjust(hspace=0.5) 
plt.savefig(os.path.join(save_path,f"b{N}_gamma_d.png"))
plt.close()

print("=============================================================")
#delta

delta_d={}
for n in range(N):#for bootstraps
    delta_d[n]={k:d_combat_output[n][k]['delta_star'] for k in d_combat_output[n].keys()}
    delta_d[n]=pd.DataFrame(delta_d[n]).T

delta_n = {i: pd.DataFrame(n_combat_output[i]['delta_star']) for i in range(N)}  
#delta is unque for each feature for each batches

p1=0.1
p2=0.9
#estimate the accumulated difference between n_delta, d_delta and average of delta_n, delta_d.
delta_d1={}
delta_d_avg={}#avg for each batch each feature

delta_d_ci_low = {}
delta_d_ci_high = {}

for b in range(n_delta.shape[0]): # for each batch
    delta_d1[b]={n:delta_d[n].iloc[b,:] for n in range(N)}
    delta_d1[b]=pd.DataFrame(delta_d1[b]).T#10 x 16

    delta_d_ci_low[b] = delta_d1[b].quantile(p1)
    delta_d_ci_high[b] = delta_d1[b].quantile(p2)

    delta_d_avg[b]=delta_d1[b].mean(axis=0)

delta_d_avg=pd.DataFrame(delta_d_avg).T #batches x features

delta_n1={}
delta_n_avg={}#avg for each batch each feature

delta_n_ci_low = {}
delta_n_ci_high = {}

for b in range(n_delta.shape[0]): # for each batch
    delta_n1[b]={n:delta_n[n].iloc[b,:] for n in range(N)}
    delta_n1[b]=pd.DataFrame(delta_n1[b]).T#10 x 16

    delta_n_ci_low[b] = delta_n1[b].quantile(p1)
    delta_n_ci_high[b] = delta_n1[b].quantile(p2)

    delta_n_avg[b]=delta_n1[b].mean(axis=0)

delta_n_avg=pd.DataFrame(delta_n_avg).T #batches x features

#=============================================================================#
#assign color for batches
num_features=len(feature_name)
keys = list(d_combat_output[1].keys())
num_keys = len(keys)
cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(i / num_keys) for i in range(num_keys)]

#plot of batches for the same feature in the same plot
fig, axs = plt.subplots(16, 1, figsize=(12, 25), sharex=False)
axs = axs.flatten()

# Convert keys to consistent discrete x values
x_vals = list(range(len(keys))) 
x_labels = keys  # your actual batch IDs

#neuro-combat
for i in range(num_features):  # for each feature
    for b in range(n_delta.shape[0]):  # for each batch
        mean_val = delta_n_avg.iloc[b, i]
        low = delta_n_ci_low[b].iloc[i]
        high = delta_n_ci_high[b].iloc[i]

        axs[i].errorbar(
            x_vals[b], mean_val,
            yerr=[[mean_val - low], [high - mean_val]],
            fmt='o', color=colors[b], capsize=3,
            label=f'Batch {keys[b]}' if i == 0 else ""
        )

        # Plot original data delta
        axs[i].scatter(x_vals[b], n_delta.iloc[b,i], color='green',marker='x',s=50)


    axs[i].set_ylabel(feature_name[i])
    axs[i].set_xticks(x_vals)
    axs[i].set_xticklabels(x_labels, rotation=45)
    axs[i].grid(True)

axs[-1].set_xlabel("Batch ID")
axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
fig.subplots_adjust(hspace=0.5) 
plt.savefig(os.path.join(save_path,f"b{N}_delta_n.png"))
plt.close()


#plot of batches for the same feature in the same plot
fig, axs = plt.subplots(16, 1, figsize=(12, 25), sharex=False)
axs = axs.flatten()

# Convert keys to consistent discrete x values
x_vals = list(range(len(keys))) 
x_labels = keys  # your actual batch IDs

#d-combat
for i in range(num_features):  # for each feature
    for b in range(n_delta.shape[0]):  # for each batch
        mean_val = delta_d_avg.iloc[b, i]
        low = delta_d_ci_low[b].iloc[i]
        high = delta_d_ci_high[b].iloc[i]

        axs[i].errorbar(
            x_vals[b], mean_val,
            yerr=[[mean_val - low], [high - mean_val]],
            fmt='o', color=colors[b], capsize=3,
            label=f'Batch {keys[b]}' if i == 0 else ""
        )
        # Plot original data delta
        axs[i].scatter(x_vals[b], d_delta.iloc[b,i], color='green', marker='x',s=50)

    axs[i].set_ylabel(feature_name[i])
    axs[i].set_xticks(x_vals)
    axs[i].set_xticklabels(x_labels, rotation=45)
    axs[i].grid(True)

axs[-1].set_xlabel("Batch ID")
axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
os.makedirs(save_path,exist_ok=True)
plt.savefig(os.path.join(save_path,f"b{N}_delta_d.png"))
plt.close()
