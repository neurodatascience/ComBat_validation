from scipy import stats
import numpy as np
import pandas as pd
from Feature_covariate_simulation_helper import age_sex_simulation,sites_samples,fixed_effect 
import os
import torch
"""Global setup"""
# np.random.seed(222)
#Total sample size (sum of all sites) 
N=1000
#the number of features 
G=2
#sites
I=4

sampling_type= 'Heterogeneity'#'Homogeneous'
age_type= 'Homogeneous'
effect_type="linear"
results_common_path=os.path.join('/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites',f'{sampling_type}_{age_type}_{effect_type}_N{N}_G{G}_I{I}')
os.makedirs(results_common_path,exist_ok=True)
#This section will be the same for both the homogeneous and heterogeneous versions.
"""Uniform distributions"""
alpha_G = np.random.uniform(0, 0.5,G)

Y_I=np.random.uniform(0,0.1,I)

"""Inverse Gamma"""
#version 1
tau_I=stats.invgamma.rvs(a=2, scale=8,size=I)#use 2 because we want bigger site effects
df1 = pd.DataFrame({'Y_i': Y_I, 'tau_i': tau_I})
gamma_IG=[]
for i in range(I):
    x=df1.iloc[i,:]
    v=stats.norm.rvs(loc=x['Y_i'], scale=(x['tau_i']), size=G)
    gamma_IG.append(v)
gamma_IG1=pd.DataFrame(np.column_stack(gamma_IG))
gamma_IG1.to_csv(os.path.join(results_common_path,"gamma_IG.csv"),index=False)
"""Gamma distributions"""
lambda_I=stats.gamma.rvs(a=50,scale=50,size=I)
v_I=stats.gamma.rvs(a=50,scale=1,size=I)
df1=pd.DataFrame({'lambda_i':lambda_I,'v_i':v_I})
delta_IG = []
for i in range(I):
    x = df1.iloc[i, :]
    v = stats.invgamma.rvs(a=(x['lambda_i'] * x['v_i']), scale=x['v_i'], size=G)
    delta_IG.append(v)
delta_IG1=pd.DataFrame(np.column_stack(delta_IG))
delta_IG1.to_csv(os.path.join(results_common_path,"delta_IG.csv"),index=False)
"""Half-Cauchy distribution with loc 0 and scale 0.2"""
sigma_G = stats.halfcauchy.rvs(loc=0,scale=0.2, size=G)

print('Random parameters from distributions: done')
##########################################################################################
#need to be standardized within site
n_samples=sites_samples(sampling_type,N,I)
script_dir=os.path.realpath(os.path.dirname(__file__))
n_samples1=pd.DataFrame(n_samples)
n_samples1.to_csv(os.path.join(results_common_path,'n_samples.csv'),index=False)
print('sample:',sum(n_samples))

age,sex=age_sex_simulation(sampling_type,age_type,n_samples,I)
standardized_age = [(a - np.mean(a)) / np.std(a) for a in age]

x_all = []
batch_pos_list = []
#add age,sex, and bach id to be in a dataset
for site in range(I):
    x_df = pd.DataFrame({'age': standardized_age[site], 'sex': sex[site]}) 
    x_all.append(x_df)
    batch_pos_list.append(np.repeat(site, x_df.shape[0]))  

x_all = pd.concat(x_all)

batch_pos = pd.DataFrame(np.concatenate(batch_pos_list), columns=['batch_pos'])  
# print(x_all.shape)
torch.manual_seed(123)
phi=[]
for g in range(G):
    phi.append(fixed_effect(x_all,effect_type))
print("phi done!")
# print(phi[0].shape)

print('==========================================================================================================')
"""Generate data"""
data=[]
epsilon=[]
for site in range(I):
    gamma_i = gamma_IG[site]
    delta_i = delta_IG[site]  
    x_df = pd.DataFrame({'age': standardized_age[site], 'sex': sex[site]})  
    data1 = pd.DataFrame({
    'batch': np.repeat(site+1, n_samples[site]),
    'age': age[site],
    'sex': sex[site]
})
    loc = np.where(batch_pos['batch_pos'].to_numpy() == site)[0]
    Feature=[]
    Ground_truth=[]
    epsilon1=[]
    for feature in range(G):
        sigma_g = sigma_G[feature]
        alpha_g=alpha_G[feature]
        gamma_ig=gamma_i[feature]
        delta_ig=delta_i[feature]
        # epsilon_ijg=stats.norm.rvs(loc=0,scale=sigma_g,size=n_samples[site])
        phi_g=phi[feature]
        phi_ig= phi_g[loc] 
        # print(len(phi_ig))

        std=delta_ig*sigma_g
        # y_ig=alpha_g+phi+gamma_ig+delta_ig*epsilon_ijg
        # theta_g=theta_G[feature]
        # ground_truth=alpha_g+np.dot(x_df,theta_g)+gamma_ig
        ground_truth=alpha_g+phi_ig
        mu=alpha_g+phi_ig+gamma_ig
        # print(len(ground_truth))
        # print(std)
        y_ig = []
        e = []

        for j in range(n_samples[site]):
            phi_ijg=phi_ig[j]
            epsilon_ijg = stats.norm.rvs(loc=0, scale=sigma_g, size=1)  
            y_ijg = alpha_g + phi_ijg + gamma_ig + delta_ig * epsilon_ijg  
            
            y_ig.append(y_ijg)  
            e.append(epsilon_ijg)  

        e = np.array(e).flatten() + 5
        y_ig = np.array(y_ig).flatten() + 5

        epsilon1.append(e)
        Feature.append(y_ig)
        Ground_truth.append(ground_truth)  
        # print(sigma_g,alpha_g,gamma_ig,delta_ig,phi_ig,theta_g)
    epsilon1=pd.DataFrame(np.column_stack(epsilon1))
    Feature=pd.DataFrame(np.column_stack(Feature),columns=[f'feature {i}' for i in range(G)])

    Ground_truth=pd.DataFrame(np.column_stack(Ground_truth),columns=[f'ground_truth {i}' for i in range(G)])
    data1=pd.concat([data1,Feature,Ground_truth],axis=1).reset_index(drop=True)
    # print(data1.shape)
    epsilon.append(epsilon1)
    data.append(data1)

epsilon=pd.concat(epsilon)
epsilon.columns=[f'epsilon {i}' for i in range(G)]
data=pd.concat(data)
data=pd.concat([data,epsilon],axis=1).reset_index(drop=True)
print("epsilon.shape:",epsilon.shape)
print("data.shape:",data.shape)

print("done")

output_dir = os.path.join(script_dir, 'simulated_data')#this directory is different with previous directory
os.makedirs(output_dir, exist_ok=True)
# epsilon.to_csv(os.path.join(output_dir, f'epsilon_{sampling_type}_age{age_type}_fixed{effect_type}_N{N}_G{G}_I{I}.csv'), index=False)
data.to_csv(os.path.join(output_dir, f'data_{sampling_type}_age{age_type}_fixed{effect_type}_N{N}_G{G}_I{I}.csv'), index=False)
