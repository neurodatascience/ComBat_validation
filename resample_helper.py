import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation")
import Parameter_estimations.neuroCombat as nc
import Parameter_estimations.distributedCombat as dc
import pandas as pd
import numpy as np
import pickle
import os
def dat_function(Data,data_type):
    if data_type=="simulated":
        dat_columns=[name for name in Data.columns if "feature" in name]
        dat=pd.DataFrame(np.array(Data[dat_columns]).T)
    elif data_type=="not_simulated":
        dat=pd.DataFrame(np.array(Data.drop(columns=["batch","age","sex"]).T))
    print("dat.shape:",dat.shape)
    return(dat)


def d_combat(data,file_path):
    #data: data with age, sex, batch and features
    #file_path: path to sotre data
    data_type="not_simulated"
    dat=dat_function(data,data_type)
    mod=pd.DataFrame(np.array(data[["age","sex"]]))
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
        f = f"{file_path}/site_out_" + str(b) + ".pickle"
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
        f = f"{file_path}/site_out_" + str(b) + ".pickle"
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
        f = f"{file_path}/site_out_" + str(b) + ".pickle"
        out = dc.distributedCombat_site(df, bat, x, central_out=central, file=f)
        site_outs.append(f)

    sites=[]
    for i, site_out in enumerate(site_outs):
        file_name = os.path.join(file_path, site_out)
        with open(file_name, "rb") as f:
            site_data = pickle.load(f)  
            sites.append(site_data)
            

    return sites


def neuro_combat(data):
    feature_cols = [col for col in data.columns if col not in ["batch", "age", "sex"]]
    dat = data[feature_cols].T
    print("dat.shape:",dat.shape)
    mod=pd.DataFrame(np.array(data[["age","sex"]]))
    print("mod.shape:",mod.shape)

    batch = data['batch'].astype(int)
    batch = pd.Categorical(batch)  
    batch_col = "batch"
    covars = pd.DataFrame({batch_col: batch})
    print("covars:",covars.shape)

    output=nc.neuroCombat(dat, covars, batch_col)
    return output
import scipy.stats as stats
def bootstrap_with_noise(data,n,sigma_hat,var_type):
    #data: input data with age, sex, batch and features
    #n:resample size per sex
    #sigma_hat: estimated variance from original data
    #var_type:the type of sigma, if sigma is esitmated by variance of feature or variance of each feature within each sex within each batch

    bootstrap_data_1=[]
    ids=data['batch'].unique()
    sex=data['sex'].unique()
    feature_cols=[col for col in data.columns if col not in ["batch", "age", "sex"]]
    if var_type=="feature":
        for j,b in enumerate(ids):
            for s in sex:
                data_sub=data[(data['batch']==b)&(data['sex']==s)].reset_index(drop=True)
                N=data_sub.shape[0] 
                draws=np.random.randint(0,N,size=n)
                bootstrap_data1=data_sub.iloc[draws, :].reset_index(drop=True)
                for i,feature in enumerate(feature_cols):
                    bootstrap_data1[feature]=bootstrap_data1[feature]+stats.norm.rvs(loc=0,scale=np.sqrt(sigma_hat[i]),
                                                                                                    size=len(bootstrap_data1[feature]))

                bootstrap_data_1.append(bootstrap_data1)
    if var_type=="not_feature":
        for j,b in enumerate(ids):
            for s in sex:
                data_sub=data[(data['batch']==b)&(data['sex']==s)].reset_index(drop=True)
                N=data_sub.shape[0] 
                draws=np.random.randint(0,N,size=n)
                bootstrap_data1=data_sub.iloc[draws, :].reset_index(drop=True)
                for i,feature in enumerate(feature_cols):
                    bootstrap_data1[feature]=bootstrap_data1[feature]+stats.norm.rvs(loc=0,scale=np.sqrt(sigma_hat[b][s][i]),
                                                                                                        size=len(bootstrap_data1[feature]))

                bootstrap_data_1.append(bootstrap_data1)

    bootstrap_data_1=pd.concat(bootstrap_data_1)
    bootstrap_data_1.columns=data.columns
    print("bootstrap_data_1.shape:",bootstrap_data_1.shape)
    
    return(bootstrap_data_1)