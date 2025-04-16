"""
This script simulates data following the method described in Figure 2 of
Silva et al. (2023) — "Fed-ComBat: A Generalized Federated Framework for
Batch Effect Harmonization in Collaborative Studies".

This script allows to run more than 1 simulation at a time.

Note:
In that preprint, they used the Gamma distribution for delta_ig.
Based on the assumption of empirical bayesian (Johnson et al., 2007),  
I let delta follow an inverse gamma distribution.
This implementation uses the inverse gamma accordingly. 
This approach is similar to that used in Hoang et al., 2024.

Simulation features:
- Supports both equal and unequal sample sizes per batch.
- Allows for balanced or imbalanced sex ratios across batches.
- Age distributions can be identical or differ across batches.
- Fixed effects from age and sex on the response variable can be linear or nonlinear.

Configuration file: simulation.json
- sampling_type: "H" (Homogeneous) or "In" (Heterogeneous)
    For homo, batch size is same across batches.
    For heterogeneous, batch size is unequal.

- simulation_times: an integer indicates the number of simulations

- sex_type: "H" (Homogeneous) or "In" (Heterogeneous)
    Determines if sex ratio is 0.5.
    
- age_type: "H" (Homogeneous) or "In" (Heterogeneous)
    Determines if the age distribution is the same across all batches or varies between them.

- effect_type: "linear" or "nonlinear"
    Specifies the type of fixed effect relationship between age/sex and the response variable.

- N: (int)
    Number of subjects per batch (if sampling_type is Homogeneous), or base size otherwise.

- G: (int)
    Number of batches (e.g., sites or scanners).

- I: (int)
    Number of features or response variables per subject.

- gamma_scale: (float)
    Scaling factor for the inverse gamma distribution used in simulating tau_I.

- smallest_sample_size_of_batches: (int)
    The size of the smallest batch.

The simulated dataset includes the ground truth (values without gamma and delta * epsilon), epsilon, batch ID, age, and sex.
"""

from scipy import stats
import numpy as np
import pandas as pd
from helper import age_sex_simulation,sites_samples,fixed_effect 
import os
import json
import time
import pickle
# np.random.seed(666)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to simulation config JSON file")
args = parser.parse_args()

parameter_path = args.config
with open(parameter_path, "r") as f:
    config = json.load(f)

#######################################################################################
# Access values
store_folder=config["store_folder"]
sampling_type = config["sampling_type"]
simulation_times = config["simulation_times"]
sex_type=config["sex_type"]
age_type = config["age_type"]
effect_type = config["effect_type"]
N = config["N"]#total sample size
G = config["G"]
I = config["I"]
gamma_scale = config["gamma_scale"]
#####################################################################################
file_name=f'N{N}'

default_path=f'/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{store_folder}'

script_dir=os.path.realpath(os.path.dirname(__file__))

max_retries = 3
# print("==========================================================================")
for i in range(simulation_times):
    print(f"simulation {i}")
    # print("n_samples done")
    n_samples=sites_samples(sampling_type,N,I)

    smallest_sample_size= int(np.min(n_samples))
    
    #****************************************************************************#
    # Add to config and save updated JSON
    config["smallest_sample_size_of_batches"] = smallest_sample_size

    with open(parameter_path, "w") as f:
        json.dump(config, f, indent=4)
    ###########************************************************##################

    results_common_path=os.path.join(default_path,
                                    f'{file_name}',f"simulation_{i}")
    os.makedirs(results_common_path,exist_ok=True)

    # Save updated config to new JSON file in result_common_path
    new_config_path = os.path.join(results_common_path, "simulation.json")
    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4)
    #*************************************************************************************#

    #This section will be the same for both the homogeneous and heterogeneous versions.
    # print("========================================================")
    # alpha_G 
    alpha_G = pd.Series(np.random.uniform(0, 0.5,G))
    # alpha_G.to_csv(os.path.join(results_common_path,'alpha_G.csv'),index=False)
    # ===============================================================================
    # gamma_IG done
    Y_I=np.random.uniform(0,0.1,I)
    tau_I=stats.invgamma.rvs(a=2, scale=gamma_scale,size=I)#site effect
    
    gamma_IG=stats.norm.rvs(loc=Y_I[:,None], scale=tau_I[:,None], size=(I, G))
    gamma_IG1 = pd.DataFrame(gamma_IG, index=[f'batch_{i}' for i in range(I)],
                  columns=[f'feature_{g}' for g in range(G)])

    # ==========================================================================
    # delta_IG 
    lambda_I=stats.gamma.rvs(a=50,scale=50,size=I)
    v_I=stats.gamma.rvs(a=50,scale=1,size=I)
    
    delta_IG = stats.invgamma.rvs(a=(lambda_I * v_I)[:,None], scale=v_I[:,None], size=(I, G))
    delta_IG1 = pd.DataFrame(delta_IG, index=[f'batch_{i}' for i in range(I)],
                  columns=[f'feature_{g}' for g in range(G)])

    # ===============================================================================
    
    sigma_G = stats.halfcauchy.rvs(loc=0,scale=0.2, size=G)

    # Random parameters from distributions: done
    ##########################################################################################
    # =======================================================================
    # age and sex 
    age,sex=age_sex_simulation(sex_type,age_type,n_samples)

    x_all = []
    batch_pos_list = []
    #add age,sex, and bach id to be in a dataset
    for site in range(I):
        x_df = pd.DataFrame({'age': age[site], 'sex': sex[site]}) 
        x_all.append(x_df)
        batch_pos_list.append(np.repeat(site, x_df.shape[0]))  

    x_all = pd.concat(x_all)
    # ==========================================================================
    batch_pos = np.concatenate(batch_pos_list) 
    
    # torch.manual_seed(123)
    phi=[]
    for g in range(G):
        phi.append(fixed_effect(x_all,effect_type))

    phi1=pd.DataFrame(phi).reset_index(drop=True)
    # phi1.to_csv(os.path.join(results_common_path,"fixed_effects.csv"),index=False)
    
    params = {
        "sample size per batch":{"data": n_samples, "columns": ["sample_size"]},
        "alpha": alpha_G.to_frame().to_dict(orient="split"),       
        "gamma": gamma_IG1.to_dict(orient="split"),                
        "delta": delta_IG1.to_dict(orient="split"),                
        "fixed_effects": phi1.to_dict(orient="split")              
    }
    #create a file containing parameters alpha,gamma,...

    for attempt in range(max_retries):
        try:
            path1=os.path.join(results_common_path,"params.pickle")
            with open(path1, "wb") as f:
                pickle.dump(params, f)
            with open(os.path.join(results_common_path, "params.json"), "w") as jf:
                json.dump(params, jf, indent=4)

            break  # success, exit loop
        except TimeoutError:
            print(f"Retrying config write... attempt {attempt+1}")
            time.sleep(2)
    else:
        raise TimeoutError(f"Failed to write config after {max_retries} attempts.")
        
    # print('==========================================================================================================')
    """Generate data"""
    data=[]
    # epsilon=[]
    for site in range(I):
        gamma_i = gamma_IG[site]
        delta_i = delta_IG[site]  
        x_df = pd.DataFrame({'age': age[site], 'sex': sex[site]})  
        data1 = pd.DataFrame({
        'batch': batch_pos_list[site],
        'age': age[site],
        'sex': sex[site]
    })
        loc = np.where(batch_pos == site)[0]
        Feature=[]
        Ground_truth=[]
        # epsilon1=[]
        for feature in range(G):
            sigma_g = sigma_G[feature]
            alpha_g=alpha_G[feature]
            gamma_ig=gamma_i[feature]
            delta_ig=delta_i[feature]
            
            phi_g=phi[feature]
            phi_ig= phi_g[loc] 

            std=delta_ig*sigma_g
 
            ground_truth=alpha_g+phi_ig
            mu=alpha_g+phi_ig+gamma_ig

            epsilon_ig = stats.norm.rvs(loc=0, scale=sigma_g, size=n_samples[site])
            y_ig = alpha_g + phi_ig + gamma_ig + delta_ig * epsilon_ig

            # epsilon1.append(epsilon_ig)
            Feature.append(y_ig)
            Ground_truth.append(ground_truth)  
            # print(sigma_g,alpha_g,gamma_ig,delta_ig,phi_ig,theta_g)
        # epsilon1=np.column_stack(epsilon1)
        
        Feature=np.column_stack(Feature)

        Ground_truth=np.column_stack(Ground_truth)

        feature_cols = [f'feature {i}' for i in range(G)]
        gt_cols = [f'ground_truth {i}' for i in range(G)]

        feature_df = pd.DataFrame(Feature, columns=feature_cols)
        gt_df = pd.DataFrame(Ground_truth, columns=gt_cols)
        data1 = pd.concat([data1, feature_df, gt_df], axis=1).reset_index(drop=True)

        # epsilon.append(epsilon1)
        data.append(data1)

    # epsilon = np.vstack(epsilon)
    # epsilon_df = pd.DataFrame(epsilon, columns=[f'epsilon {i}' for i in range(G)])

    data = pd.concat(data).reset_index(drop=True)
    # data = pd.concat([data, epsilon_df], axis=1).round(5)
    print(data.shape)
    
    # print("===============================================================================")
    #round data to be 5 decimals
    data=data.round(5)

    data.to_csv(os.path.join(results_common_path, 'data.csv'), index=False)#data_{sampling_type}_age{age_type}_fixed{effect_type}_N{N}_G{G}_I{I}.csv'
    



