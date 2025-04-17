"""
This script contains all the functions used in other scripts, 
with a description provided before each function.
"""

import numpy as np
import pandas as pd
import neuroCombat as nc
import distributedCombat as dc
import os
import pickle
import re
import seaborn as sns
import itertools
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def bootstrap(data):
    """
    Perform stratified bootstrap sampling within each batch.

    Parameters:
    data (DataFrame): Must contain columns 'batch' (grouping variable), 'age', 'sex', 
                      and the features we want to resample.

    Returns:
    DataFrame: Bootstrapped dataset with hierarchical batch index.
    """
    bootstrap_data = []
    ids = data['batch'].unique()  
    
    for batch_id in ids:
        d = data[data['batch'] == batch_id].reset_index(drop=True)  
        n = d.shape[0]
        draws = np.random.randint(0, n, size=n) 
        bootstrapped_sample = d.iloc[draws, :].reset_index(drop=True)
 
        bootstrap_data.append(bootstrapped_sample)

    bootstrap_data = pd.concat(bootstrap_data, ignore_index=True)
    bootstrap_data.columns=data.columns

    return bootstrap_data 
        
def bootstrap_ntimes(data,n):
    """
    Perform n bootstrap resampling iterations.

    Parameters:
    - data (pd.DataFrame): Original dataset containing 'age', 'sex', 'batch' (ID), and features.
    - n (int): Number of bootstrap resampling iterations.

    Returns:
    - pd.DataFrame: Bootstrapped datasets stacked together, with an iteration index.
    """
    b_n_data={}
    for i in range(n):
        b_n_data[i]=bootstrap(data)
    
    return b_n_data

def neuro_combat_train(bootstrapped_data):
    """
    Perform neuroCombat batch correction on bootstrapped datasets.

    Parameters:
    - bootstrapped_data (dict): Dictionary where each key contains a DataFrame from one bootstrap iteration.

    Returns:
    - dict: Dictionary containing batch-corrected data and estimated parameters for each bootstrap iteration.
    """
    if isinstance(bootstrapped_data, dict):
        ntimes=len(bootstrapped_data)#number of bootstrap done
        
        result={}
        for t in range(ntimes):
            data=bootstrapped_data[t].copy() 

            #ensure their class is matching the requirement of model
            data['age'] = pd.to_numeric(data['age'], errors='coerce')
            data['sex'] = data['sex'].astype('category') 
          
            #do neuro-combat for this resampled data
            feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
            dat = data[feature_cols].T
            # print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            # print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch,
                                   mod.columns[0]:mod[mod.columns[0]],
                                   mod.columns[1]:mod[mod.columns[1]]})

            # print("covars:",covars.columns)

            output=nc.neuroCombat(dat,covars, batch_col,categorical_cols='sex',continuous_cols='age')
            result[t]={"combat_data":output['data'],
                    "alpha":output["estimates"]['stand.mean'],
                    "beta":output["estimates"]['beta.hat'][-2:,],
                    "XB":output['estimates']['mod.mean'],
                    "sigma": output['estimates']['var.pooled'],
                    "delta_star":output['estimates']['delta.star'],
                    "gamma_star":output['estimates']['gamma.star']}
    else:
            data=bootstrapped_data.copy()
            data['age'] = pd.to_numeric(data['age'], errors='coerce')
            data['sex'] = data['sex'].astype('category') 

            #do neuro-combat for this resampled data
            feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
            dat = data[feature_cols].T
            # print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            # print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch,mod.columns[0]:mod[mod.columns[0]],
                                   mod.columns[1]:mod[mod.columns[1]]})
            # print("covars:",covars.shape)

            output=nc.neuroCombat(dat, covars, batch_col,categorical_cols='sex',continuous_cols='age')
            result={"combat_data":output['data'],
                    "alpha":output["estimates"]['stand.mean'],
                    "beta":output["estimates"]['beta.hat'][-2:,],#last two rows are beta for age and sex
                    "XB":output['estimates']['mod.mean'],
                    "sigma": output['estimates']['var.pooled'],
                    "delta_star":output['estimates']['delta.star'],
                    "gamma_star":output['estimates']['gamma.star']}

    return result

def d_combat_train(bootstrapped_data,file_path):
    """
    Perform neuroCombat batch correction on bootstrapped datasets.

    Parameters:
    - bootstrapped_data (dict): Dictionary where each key contains a DataFrame from one bootstrap iteration.
    - file_path: the local directory you want to save site data.
    Returns:
    - dict: Dictionary containing batch-corrected data and estimated parameters for each bootstrap iteration.
    """
    if isinstance(bootstrapped_data, dict):
        ntimes=len(bootstrapped_data)#number of bootstrap done
        
        result={}
        for t in range(ntimes):
            result[t]={}
            file_path1=f"{file_path}/bootstrap_{t}"
            os.makedirs(file_path1,exist_ok=True)
            data=bootstrapped_data[t]            
            #do neuro-combat for this resampled data
            feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
            dat = pd.DataFrame(np.array(data[feature_cols]).T)
            # print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            # print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch})
            # print("covars:",covars.shape)

            print("Step 1")
            site_outs = []
            for b in covars[batch_col].unique():
                s = covars[batch_col] == b  
                df = dat.loc[:, s.to_numpy()]  
                bat = covars[batch_col][s]
                x = mod.loc[s, :]
                f = f"{file_path1}/site_out_" + str(b) + ".pickle"
                out = dc.distributedCombat_site(df, bat, x, verbose=True, file=f)
                site_outs.append(f)

            central = dc.distributedCombat_central(site_outs)

            # print("central['var_pooled']:", central['var_pooled'])
            print("=================================================================")
            ### Step 2
            print("Step 2")
            site_outs = []
            for b in covars[batch_col].unique():
                s = covars[batch_col] == b  
                df = dat.loc[:, s.to_numpy()]  
                bat = covars[batch_col][s]
                x = mod.loc[s, :]
                f = f"{file_path1}/site_out_" + str(b) + ".pickle"
                out = dc.distributedCombat_site(df, bat, x, verbose=True, central_out=central, file=f)
                site_outs.append(f)

            central = dc.distributedCombat_central(site_outs)
            # print("central['var_pooled']:", central['var_pooled'])

            print("Step 3")
            site_outs = []
            for b in covars[batch_col].unique():
                s = list(map(lambda x: x == b, covars[batch_col]))
                df = dat.loc[:, s]
                bat = covars[batch_col][s]
                x = mod.loc[s, :]
                f = f"{file_path1}/site_out_" + str(b) + ".pickle"
                out = dc.distributedCombat_site(df, bat, x, central_out=central, file=f)
                site_outs.append(f)

            # print(site_outs)
            for i, site_out in enumerate(site_outs):
                file_name = site_out
                with open(file_name, "rb") as f:
                    site_data = pickle.load(f)  
                    digit=int(re.findall(r'\d+', site_out)[-1])
                    # print(digit)
                    result[t][digit]={"combat_data":site_data['dat_combat'],
                    "alpha":site_data["estimates"]['stand_mean'],
                    "beta":site_data["estimates"]['beta_hat'][-2:,][::-1],#switch the first row to be the second, match the order of covariates in neuro-combat
                    "XB":site_data['estimates']['mod_mean'],
                    "sigma": site_data['estimates']['var_pooled'],
                    "delta_star":site_data['estimates']['delta_star'],
                    "gamma_star":site_data['estimates']['gamma_star']}
                # print("done!")

    else:
            file_path1=f"{file_path}/origin"
            os.makedirs(file_path1,exist_ok=True)
            data=bootstrapped_data
            #do neuro-combat for this resampled data
            feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
            dat = pd.DataFrame(np.array(data[feature_cols]).T)
            # print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            # print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch})
            # print("covars:",covars.shape)

            print("Step 1")
            site_outs = []
            for b in covars[batch_col].unique():
                s = covars[batch_col] == b  
                df = dat.loc[:, s.to_numpy()]  
                bat = covars[batch_col][s]
                x = mod.loc[s, :]
                f = f"{file_path1}/site_out_" + str(b) + ".pickle"
                out = dc.distributedCombat_site(df, bat, x, verbose=True, file=f)
                site_outs.append(f)

            central = dc.distributedCombat_central(site_outs)

            # print("central['var_pooled']:", central['var_pooled'])
            print("=================================================================")
            ### Step 2
            print("Step 2")
            site_outs = []
            for b in covars[batch_col].unique():
                s = covars[batch_col] == b  
                df = dat.loc[:, s.to_numpy()]  
                bat = covars[batch_col][s]
                x = mod.loc[s, :]
                f = f"{file_path1}/site_out_" + str(b) + ".pickle"
                out = dc.distributedCombat_site(df, bat, x, verbose=True, central_out=central, file=f)
                site_outs.append(f)

            central = dc.distributedCombat_central(site_outs)
            # print("central['var_pooled']:", central['var_pooled'])

            print("Step 3")
            site_outs = []
            for b in covars[batch_col].unique():
                s = list(map(lambda x: x == b, covars[batch_col]))
                df = dat.loc[:, s]
                bat = covars[batch_col][s]
                x = mod.loc[s, :]
                f = f"{file_path1}/site_out_" + str(b) + ".pickle"
                out = dc.distributedCombat_site(df, bat, x, central_out=central, file=f)
                site_outs.append(f)

            """beta from nuero-combat and d-combat are not in the same order."""
            """neuro-combat: sex, age and d-combat: age,sex"""

            result={}
            for i, site_out in enumerate(site_outs):
                file_name = site_out
                with open(file_name, "rb") as f:
                    site_data = pickle.load(f) 
                    digit=int(re.findall(r'\d+', site_out)[-1])
                    # print(digit)
                    result[digit]={"combat_data":site_data['dat_combat'],
                    "alpha":site_data["estimates"]['stand_mean'],
                    "beta":site_data["estimates"]['beta_hat'][-2:,][::-1],#switch the first row to be the second, match the order of covariates in neuro-combat
                    "XB":site_data['estimates']['mod_mean'],
                    "sigma": site_data['estimates']['var_pooled'],
                    "delta_star":site_data['estimates']['delta_star'],
                    "gamma_star":site_data['estimates']['gamma_star']}


    return result


def ci_plot(df,y_label,save_path,confidence,y):
    """
    Plots RMSE with a 95% confidence interval shaded area.
    
    Parameters:
        df (pd.DataFrame): Must contain 'Sample Size', 'RMSE', 'Lower Bound', 'Upper Bound'
        y_label (str): Label for the y-axis
        save_path (str): Directory to save the plot
        y: variable look as y
    """

    plt.figure(figsize=(20, 8))

    # Main RMSE line plot
    sns.lineplot(x='Sample Size', y=y, data=df, marker='o', label=y)

    # Shaded confidence interval
    plt.fill_between(df['Sample Size'], df['Lower Bound'], df['Upper Bound'], 
                        color='pink', alpha=0.5, label=f'{confidence}% Confidence Interval')
    
    plt.xlabel("Sample Size")
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig(os.path.join(save_path,f"{y_label}.png"))
    plt.close()


def harmonized_plot(data,neuro_combat,file_path,png_name):
    """
    This function compares harmonized data with non-harmonized data and the ground truth.
    Each row corresponds to a different feature.

    - The first column shows the ground truth.
    - The second column displays the non-harmonized data.
    - The third column presents the harmonized data.
    """

    G=len([col for col in data.columns if "feature" in col])
    
    fig, axes = plt.subplots(G,3, figsize=(5*G, 6)) 

    # Define color cycle
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    unique_combinations = [(b, s) for b in data['batch'].unique() for s in np.unique(data['sex'])]
    color_map = cm.get_cmap('tab10', len(unique_combinations))
    color_dict = {combo: color_map(i) for i, combo in enumerate(unique_combinations)}


    legend_entries = {}

    for i in range(G): 
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        feature_name = f'feature {i}'
        ground_name = f'ground_truth {i}'

        # First pass to determine axis limits
        for batch in data['batch'].unique():
            s=np.where(data['batch']==batch)[0]
            d = data.iloc[s,]
            age = d['age']
            ground = d[ground_name]  
            y = d[feature_name]  
            
            y_n = neuro_combat.iloc[s, i]
  
            x_min = min(x_min, age.min())-5
            x_max = max(x_max, age.max())+5

            y_min = min(y_min, ground.min(), y.min(), y_n.min())-2
            y_max = max(y_max, ground.max(), y.max(), y_n.max())+2

        
        for batch in data['batch'].unique():
            # Second pass to plot
            s=np.where(data['batch']==batch)[0]
            d = data.iloc[s,]
            age = d['age']
            ground = d[ground_name]  
            y = d[feature_name]  
            current_sex = d['sex'].values  

            y_n = neuro_combat.iloc[s, i] 

            unique_sexes = np.unique(current_sex)

            for s in unique_sexes:
                indices = np.where(current_sex == s)[0]
                color = color_dict[(batch, s)]  # Get color from the dictionary

                row = i
                col = 0  

                y_min1=ground.iloc[indices].min()
                y_max1=ground.iloc[indices].max()
                # Ground-truth plot
                ax = axes[row, col]
                scatter = ax.scatter(age.iloc[indices], ground.iloc[indices], label=f'batch {batch}, sex {s}', s=8, color=color)
                ax.set_title(f'Ground Truth - {feature_name}')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.axhline(y=y_min1, color='grey', linestyle='--', linewidth=1,alpha=0.2)
                ax.axhline(y=y_max1, color='grey', linestyle='--', linewidth=1,alpha=0.2)

                if f'batch {batch}, sex {s}' not in legend_entries:
                    legend_entries[f'batch {batch}, sex {s}'] = scatter


                y_min2=y.iloc[indices].min()
                y_max2=y.iloc[indices].max()

                # Non-harmonized plot
                col = 1
                ax = axes[row, col]
                ax.scatter(age.iloc[indices], y.iloc[indices], s=8, color=color)
                ax.set_title(f'Non-Harmonized - {feature_name}')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.axhline(y=y_min2, color='grey', linestyle='--', linewidth=1,alpha=0.2)
                ax.axhline(y=y_max2, color='grey', linestyle='--', linewidth=1,alpha=0.2)


                y_min3=y_n.iloc[indices].min()
                y_max3=y_n.iloc[indices].max()
                # Neuro-combat plot
                col = 2
                ax = axes[row, col]
                ax.scatter(age.iloc[indices], y_n.iloc[indices], s=8, color=color)
                ax.set_title(f'Neuro-Combat - {feature_name}')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.axhline(y=y_min3, color='grey', linestyle='--', linewidth=1,alpha=0.2)
                ax.axhline(y=y_max3, color='grey', linestyle='--', linewidth=1,alpha=0.2)


    fig.legend(handles=legend_entries.values(), labels=legend_entries.keys(),
            loc='upper left', bbox_to_anchor=(0.85, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  
    plt.savefig(os.path.join(file_path,png_name))
    plt.close()


def sites_samples(sampling_type, N, I):
    """
    This function determines the site sizes for the simulation.

    - 'H' stands for homogeneous, where each site has an equal sample size.
    - 'In' stands for heterogeneous, where site sample sizes are unequal.
    """
    if sampling_type == 'H':
        n_i = int(N / I)
        n_samples = [n_i for _ in range(I)]
    elif sampling_type == 'In':
        alpha = np.random.randint(1, 21, size=I)
        p = np.random.dirichlet(alpha)
        n_samples = (N * p).round(0).astype(int)
    else:
        raise ValueError("Invalid sampling_type. Choose 'Homogeneous' or 'Heterogeneity'.")
    
    return n_samples

def age_sex_simulation(sex_type,age_type,n_samples):
    """
    This function simulates age and sex data.

    - 'H' stands for homogeneous: the probability of selecting a female is 0.5, and all sites share the same age distribution.
    - 'In' stands for heterogeneous: the probability of selecting a female is drawn from a Uniform(0.2, 0.8) distribution, and each site has a unique age distribution.
    """

    age=[]
    sex=[]
    
    if sex_type=="H":
        for n in n_samples:
            v1=np.random.binomial(n=1, p=0.5, size=n)
            sex.append(v1)
    else:
        p=[np.random.uniform(0.2,0.8,1) for _ in range(len(n_samples))]#each site has a unique sex distribution
        for i,n in enumerate(n_samples):
            v1=np.random.binomial(n=1, p=p[i], size=n)
            sex.append(v1)
    if age_type=='H':
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


def fixed_effect_nonlinear(x_df):
    """
    This function implements the nonlinear form of fixed effects for age and sex.
    """

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
    """
    This function generates the fixed effects of age and sex on a feature.

    It supports two effect types:
    - 'linear': a linear relationship between the covariates and the feature.
    - 'nonlinear': a nonlinear relationship between the covariates and the feature.
    """
    if effect_type=="nonlinear":
        return (fixed_effect_nonlinear(x_df))
    if effect_type=="linear":
        theta_g=(stats.norm.rvs(loc=0,scale=1,size=2))
        phi_g=np.dot(x_df,theta_g)
        return(phi_g)

            

