import numpy as np
import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation")
import Parameter_estimations.neuroCombat as nc
import pandas as pd
import Parameter_estimations.distributedCombat as dc
import os
import pickle

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
            data=bootstrapped_data[t]  

            #ensure their class is matching the requirement of model
            data['age'] = pd.to_numeric(data['age'], errors='coerce')
            data['sex'] = data['sex'].astype('category') 
          
            #do neuro-combat for this resampled data
            feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
            dat = data[feature_cols].T
            print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch,
                                   mod.columns[0]:mod[mod.columns[0]],
                                   mod.columns[1]:mod[mod.columns[1]]})

            print("covars:",covars.columns)

            output=nc.neuroCombat(dat,covars, batch_col,categorical_cols='sex',continuous_cols='age')
            result[t]={"combat_data":output['data'],
                    "alpha":output["estimates"]['stand.mean'],
                    "beta":output['estimates']['beta.hat'],
                    "delta_star":output['estimates']['delta.star'],
                    "gamma_star":output['estimates']['gamma.star']}
    else:
            data=bootstrapped_data
            data['age'] = pd.to_numeric(data['age'], errors='coerce')
            data['sex'] = data['sex'].astype('category') 

            #do neuro-combat for this resampled data
            feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
            dat = data[feature_cols].T
            print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch,mod.columns[0]:mod[mod.columns[0]],
                                   mod.columns[1]:mod[mod.columns[1]]})
            print("covars:",covars.shape)

            output=nc.neuroCombat(dat, covars, batch_col,categorical_cols='sex',continuous_cols='age')
            result={"combat_data":output['data'],
                    "alpha":output["estimates"]['stand.mean'],
                    "beta":output['estimates']['beta.hat'],
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
            print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch})
            print("covars:",covars.shape)

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

            print("central['var_pooled']:", central['var_pooled'])
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
            print("central['var_pooled']:", central['var_pooled'])

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


            for i, site_out in enumerate(site_outs):
                file_name = os.path.join(file_path1, site_out)
                with open(file_name, "rb") as f:
                    site_data = pickle.load(f)  
                    result[t][i]={"combat_data":site_data['dat_combat'],
                    "alpha":site_data["estimates"]['stand_mean'],
                    "beta":site_data['estimates']['beta_hat'],
                    "delta_star":site_data['estimates']['delta_star'],
                    "gamma_star":site_data['estimates']['gamma_star']}

    else:
            file_path1=f"{file_path}/origin"
            os.makedirs(file_path1,exist_ok=True)
            data=bootstrapped_data
            #do neuro-combat for this resampled data
            feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
            dat = pd.DataFrame(np.array(data[feature_cols]).T)
            print("dat.shape:",dat.shape)
            mod=pd.DataFrame(np.array(data[["age","sex"]]))
            mod.columns=["age","sex"]
            print("mod.shape:",mod.shape)

            batch = data['batch'].astype(int)
            batch = pd.Categorical(batch)  
            batch_col = "batch"
            covars = pd.DataFrame({batch_col: batch})
            print("covars:",covars.shape)

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

            print("central['var_pooled']:", central['var_pooled'])
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
            print("central['var_pooled']:", central['var_pooled'])

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

            result={}
            for i, site_out in enumerate(site_outs):
                file_name = os.path.join(file_path1, site_out)
                with open(file_name, "rb") as f:
                    site_data = pickle.load(f)  
                    result[i]={"combat_data":site_data['dat_combat'],
                    "alpha":site_data["estimates"]['stand_mean'],
                    "beta":site_data['estimates']['beta_hat'],
                    "delta_star":site_data['estimates']['delta_star'],
                    "gamma_star":site_data['estimates']['gamma_star']}


    return result
