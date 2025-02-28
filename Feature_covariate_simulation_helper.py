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


def age_sex_simulation(sampling_type,age_type,n_samples,I):#for sampling type being homo, age tpye must be homo, for heteroneity, we have homo and inhomo for age type
    age=[]
    sex=[]
    # n_samples=sites_samples(sampling_type, N, I)
    if sampling_type=='Homogeneous' and age_type=='Homogeneous':
        for n in n_samples:
            #covariates sex
            #for site i with n_i samples, the probability of male is equal to female as 50%.
            v1=np.random.binomial(n=1, p=0.5, size=n)
            sex.append(v1)
            #covariate age
            #by reading the median of all boxplots, I guess the mean to be 40 and 10 for sd.
            v2=stats.norm.rvs(loc=50, scale=15, size=n)
            v2 = v2[v2 > 0]
            # print(len(v2))
            while len(v2) < n:
                additional_samples = stats.norm.rvs(loc=50, scale=15, size=n - len(v2))
                v2 = np.concatenate((v2, additional_samples[additional_samples > 0]))
            age.append(v2)
    elif sampling_type == 'Heterogeneity':
        if age_type=='Homogeneous':
            p=[np.random.uniform(0.2,0.8,1) for _ in range(len(n_samples))]#each site has a unique sex distribution
            # print('p:',p)
            mu=np.random.uniform(20,65,1)
            a=mu/3-5
            b=mu/3+5
            sd=float(np.random.uniform(a, b, 1))

            for i, n in enumerate(n_samples):
                print('i,n:',i,n)
                #covariates sex
                v1=np.random.binomial(n=1, p=p[i], size=n)
                sex.append(v1)
                #covariate age
                #by reading the median of all boxplots, I guess the mean to be 40 and 10 for sd.
                v2=stats.norm.rvs(loc=mu, scale=sd, size=n)
                v2 = v2[v2 > 0]
                # print(len(v2))
                while len(v2) < n:
                    additional_samples = stats.norm.rvs(loc=mu, scale=sd, size=n - len(v2))
                    v2 = np.concatenate((v2, additional_samples[additional_samples > 0]))
                age.append(v2)

        if age_type=='Heterogeneity':
            p=[np.random.uniform(0.2,0.8,1) for _ in range(len(n_samples))]#each site has a unique sex distribution
            # print('p:',p)
            mu_age=np.random.uniform(20,65,I)

            for i, n in enumerate(n_samples):
                print('i,n:',i,n)
                #covariates sex
                v1=np.random.binomial(n=1, p=p[i], size=n)
                sex.append(v1)
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

# def smooth_leaky_relu(x, alpha=0.9, beta=10):
#     return alpha * torch.log(1+ torch.exp(beta * x)) / beta

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
            # random_val = torch.rand_like(x) * 0.5  # Random value in (0, 0.5)
            # log_value = torch.log(1 + random_val)  # Compute log(1 + a)
            epsilon=0
            # a = 2
            # b =0
            # c = 0
            # quadratic_part = a * x**2 + b * x + c  # Parabolic smoothing

            # negative_slope=-1
            leaky_relu_part = torch.where(x >epsilon, x, -x)


            return leaky_relu_part

           
        def forward(self, x):
            """make a gausian plot"""
            # leaky_relu=nn.LeakyReLU(negative_slope=-1)
            # elu = nn.ELU(alpha=1.0)
            h = torch.sigmoid(self.hidden_layer1(x))
            #torch.tanh(self.hidden_layer1(x))
            #leaky_relu(self.hidden_layer1(x))
            # h = torch.relu(self.hidden_layer2(h))
            # h = torch.relu(self.hidden_layer3(h))
            out = 10*(np.sin(2*np.pi*self.output_layer(h)/10))
            return (out)

    # torch.manual_seed(549)
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
