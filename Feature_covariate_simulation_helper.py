"""This script contains functions used to simulate data for Com-Bat and d-ComBat"""

from spicy import stats
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


def age_sex_simulation(sampling_type,N,I):
    age=[]
    sex=[]
    n_samples=sites_samples(sampling_type, N, I)
    if sampling_type=='Homogeneous':
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
                additional_samples = stats.norm.rvs(loc=40, scale=10, size=n - len(v2))
                v2 = np.concatenate((v2, additional_samples[additional_samples > 0]))
            age.append(v2)
    elif sampling_type == 'Heterogeneity':
        p=[np.random.uniform(0.2,0.8,n) for n in n_samples]
        mu_age=np.random.uniform(10,90,I)
        i=0
        for n in n_samples:
            #covariates sex
            #for site i with n_i samples, the probability of male is equal to female as 50%.
            v1=np.random.binomial(n=1, p=p[i], size=n)
            sex.append(v1)
            #covariate age
            mu=mu_age[i]
            a=mu/3-5
            b=mu/3+5
            sd=np.random.uniform(a,b,1)
            #by reading the median of all boxplots, I guess the mean to be 40 and 10 for sd.
            v2=stats.norm.rvs(loc=mu, scale=sd, size=n)
            age.append(v2)
            i+=1
    return age,sex
    
def fixed_effect_nonlinear(x_df):
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
        
        def forward(self, x):
            h = torch.relu(self.hidden_layer1(x))
            # h = torch.relu(self.hidden_layer2(h))
            # h = torch.relu(self.hidden_layer3(h))
            out = self.output_layer(h)
            return out

    torch.manual_seed(549)
    global_model = MLP(input_dim=2, hidden_dim=128, output_dim=1)
    global_model.eval()
    X_tensor = torch.tensor(x_df.values, dtype=torch.float32)
        
    with torch.no_grad():
        y_simulated = global_model(X_tensor).numpy()
    
    return y_simulated.flatten()


