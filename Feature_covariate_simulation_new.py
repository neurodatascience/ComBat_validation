from spicy import stats
import numpy as np
import pandas as pd
from Feature_covariate_simulation_helper import age_sex_simulation,sites_samples,fixed_effect_nonlinear 
import os
"""Global setup"""
#Total sample size (sum of all sites) 
N=500
#the number of features 
G=1
#10 sites
I=10
sampling_type='Homogeneous'
#This section will be the same for both the homogeneous and heterogeneous versions.
"""Uniform distributions"""
alpha_G = np.random.uniform(0, 0.5,G)

Y_I=np.random.uniform(0,0.1,I)

"""Inverse Gamma"""
#version 1
tau_I=stats.invgamma.rvs(a=2, scale=0.5,size=I)
df1 = pd.DataFrame({'Y_i': Y_I, 'tau_i': tau_I})
gamma_IG=[]
for i in range(I):
    x=df1.iloc[i,:]
    v=stats.norm.rvs(loc=x['Y_i'], scale=np.sqrt(x['tau_i']), size=G)
    gamma_IG.append(v)

"""Gamma distributions"""
lambda_I=stats.gamma.rvs(a=50,scale=1/50,size=I)
v_I=stats.gamma.rvs(a=50,scale=1,size=I)
df1=pd.DataFrame({'lambda_i':lambda_I,'v_i':v_I})
delta_IG = []
for i in range(I):
    x = df1.iloc[i, :]
    v = stats.gamma.rvs(a=(x['lambda_i'] * x['v_i']), scale=1/x['v_i'], size=G)
    delta_IG.append(v)

"""Half-Cauchy distribution with loc 0 and scale 0.2"""
sigma_G = stats.halfcauchy.rvs(loc=0,scale=0.2, size=G)

print('Random parameters from distributions: done')
##########################################################################################
#need to be standardized within site
age,sex=age_sex_simulation(sampling_type,N,I)
standardized_age = [(a - np.mean(a)) / np.std(a) for a in age]
n_samples=sites_samples(sampling_type, N, I)

data=[]
for site in range(I):
    gamma_i = gamma_IG[site]
    delta_i = delta_IG[site]  
    x_df = pd.DataFrame({'age': standardized_age[site], 'sex': sex[site]})  
    data1=pd.DataFrame({'batch':site,'age':age[site],'sex':sex[site]})
    Feature=[]
    Ground_truth=[]
    for feature in range(G):
        sigma_g = sigma_G[feature]
        alpha_g=alpha_G[feature]
        gamma_ig=gamma_i[feature]
        delta_ig=delta_i[feature]
        epsilon_ijg=stats.norm.rvs(loc=0,scale=sigma_g,size=n_samples[site])
        phi= fixed_effect_nonlinear(x_df) 
        y_ig=alpha_g+phi+gamma_ig+delta_ig*epsilon_ijg
        ground_truth=alpha_g+phi+gamma_ig
        Feature.append(y_ig)
        Ground_truth.append(ground_truth)
    Feature=pd.DataFrame(np.column_stack(Feature),columns=[f'feature {i}' for i in range(G)])

    Ground_truth=pd.DataFrame(np.column_stack(Ground_truth),columns=[f'ground truth feature {i}' for i in range(G)])
    data1=pd.concat([data1,Feature,Ground_truth],axis=1)
    data.append(data1)

data=pd.concat(data)
script_dir = os.path.realpath(os.path.dirname(__file__))
data.to_csv(os.path.join(script_dir,'simulated_data',f'data_{sampling_type}_N{N}_G{G}_I{I}.csv'),index=False)