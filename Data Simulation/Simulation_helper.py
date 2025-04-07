"""This script contains functions used to simulate data for Com-Bat and d-ComBat"""

from scipy import stats
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
np.random.seed(42)
"""Homogeneous and heterogenity"""
def sites_samples(sampling_type, N, I):
    if sampling_type == 'Homogeneous':
        n_i = int(N / I)
        n_samples = [n_i for _ in range(I)]
    elif sampling_type == 'Heterogeneity':
        alpha = np.random.randint(1, 21, size=I)
        p = np.random.dirichlet(alpha)
        n_samples = (N * p).round(0).astype(int)
    else:
        raise ValueError("Invalid sampling_type. Choose 'Homogeneous' or 'Heterogeneity'.")
    # print('n_samples: done')
    return n_samples


def age_sex_simulation(sex_type,age_type,n_samples):#for sampling type being homo, age tpye must be homo, for heteroneity, we have homo and inhomo for age type
    age=[]
    sex=[]
    
    if sex_type=="Homogenous":
        for n in n_samples:
            v1=np.random.binomial(n=1, p=0.5, size=n)
            sex.append(v1)
    else:
        p=[np.random.uniform(0.2,0.8,1) for _ in range(len(n_samples))]#each site has a unique sex distribution
        for i,n in enumerate(n_samples):
            v1=np.random.binomial(n=1, p=p[i], size=n)
            sex.append(v1)
    if age_type=='Homogeneous':
        mu=np.random.uniform(20,65,1)
        a=mu/3-5
        b=mu/3+5
        sd=float(np.random.uniform(a, b, 1))
        for i, n in enumerate(n_samples):
            #covariate age
            #by reading the median of all boxplots, I guess the mean to be 40 and 10 for sd.
            v2=stats.norm.rvs(loc=mu, scale=sd, size=n)
            v2 = v2[v2 > 0]
            # print(len(v2))
            while len(v2) < n:
                additional_samples = stats.norm.rvs(loc=mu, scale=sd, size=n - len(v2))
                v2 = np.concatenate((v2, additional_samples[additional_samples > 0]))
            age.append(v2)
    else:
        mu_age=np.random.uniform(20,65,len(n_samples))
        for i, n in enumerate(n_samples):
            #covariate age
            mu=mu_age[i]
            a=mu/3-5
            b=mu/3+5
            sd=float(np.random.uniform(a, b, 1))
            #by reading the median of all boxplots, I guess the mean to be 40 and 10 for sd.
            v2=stats.norm.rvs(loc=mu, scale=sd, size=n)
            v2 = v2[v2 > 0]
            # print(len(v2))
            while len(v2) < n:
                additional_samples = stats.norm.rvs(loc=mu, scale=sd, size=n - len(v2))
                v2 = np.concatenate((v2, additional_samples[additional_samples > 0]))
            age.append(v2)
            
    return age,sex


def u_shape(x, a=1, b=0, c=0):
    return a * (x - b) ** 2 + c

import torch.nn.functional as F   
import math 
def fixed_effect_nonlinear(x_df):#control this to have at most on modal
    class MLP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
            super(MLP, self).__init__()
            
            self.hidden_layer1 = nn.Linear(input_dim, hidden_dim)
            # self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
            # self.hidden_layer3 = nn.Linear(hidden_dim, hidden_dim)
            nn.init.normal_(self.hidden_layer1.weight, mean=0, std=1)  
            nn.init.normal_(self.hidden_layer1.bias, mean=0, std=1)    
            # nn.init.normal_(self.hidden_layer2.weight, mean=0, std=1)  
            # nn.init.normal_(self.hidden_layer2.bias, mean=0, std=1)    
            # # nn.init.normal_(self.hidden_layer3.weight, mean=0, std=1)  
            # nn.init.normal_(self.hidden_layer3.bias, mean=0, std=1)    

            
            self.output_layer = nn.Linear(hidden_dim, output_dim)
            nn.init.normal_(self.output_layer.weight, mean=0, std=1)
            nn.init.normal_(self.output_layer.bias, mean=0, std=1)
        def custom_leaky_relu(self, x):
            """Custom Leaky ReLU: Uses log(1 + random_val) when x == 0"""
            epsilon=0
            leaky_relu_part = torch.where(x >epsilon, x, -x)


            return leaky_relu_part

           
        def forward(self, x):
            h = torch.sigmoid(self.hidden_layer1(x))
            out = 10*(np.sin(2*np.pi*self.output_layer(h)/10))#transformation done here for ensuring non-negative
            return (out)

    global_model = MLP(input_dim=2, hidden_dim=128, output_dim=1)
    global_model.eval()
    X_tensor = torch.tensor(x_df.values, dtype=torch.float32)
        
    with torch.no_grad():
        y_simulated = global_model(X_tensor).numpy()
    
    return y_simulated.flatten()

def fixed_effect(x_df,effect_type):
    if effect_type=="nonlinear":
        return (fixed_effect_nonlinear(x_df))
    if effect_type=="linear":
        theta_g=(stats.norm.rvs(loc=0,scale=1,size=2))
        phi_g=np.dot(x_df,theta_g)
        return(phi_g)

            


    
