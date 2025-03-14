import numpy as np
import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation")
import Parameter_estimations.neuroCombat as nc
import pandas as pd

def neuro_combat_bootstrap_data(ids,data,n):
    #ids: batch id we are interested in 
    #data: data corresponding to ids (should only contain age, sex, batch, and features we want to harmonize)
    #n: sample size (how many samples do we want to get from one bootstrap)
    
    if len(ids)>1:#if we are interested in more than one batch
        bootstrap_data=[]
        sample_size=[]
        for ID in ids:
            data_sub = data[data["batch"] == ID].reset_index(drop=True)            
            N=data_sub.shape[0]    
            draws=np.random.randint(0,N,size=n)#draw data with replacement
            unique_samples_size=len(np.unique(draws))
            bootstrap_data1=data_sub.iloc[draws, :].reset_index(drop=True)
            bootstrap_data.append(bootstrap_data1)
            sample_size.append(unique_samples_size)
            # print("id:",id)
            # print("len(draws):",len(draws))
        bootstrap_data=pd.concat(bootstrap_data)
        bootstrap_data.columns=data.columns
        print("bootstrap_data.shape:",bootstrap_data.shape)
        # sample_size=pd.DataFrame(sample_size).to_numpy()
        # print("(sample_size):",(sample_size))
    else:
        data_sub=data[data["batch"]==ids].reset_index(drop=True)
        N=data_sub.shape[0]
        draws=np.random.randint(0,N,size=n)
        sample_size=len(np.unique(draws))
        bootstrap_data=data_sub.iloc[draws, :].reset_index(drop=True)
        bootstrap_data.columns=data.columns
        print("bootstrap_data.shape:",bootstrap_data.shape)

    #do neuro-combat for this resampled data
    feature_cols = [col for col in bootstrap_data.columns if col not in ["batch", "age", "sex"]]
    dat = bootstrap_data[feature_cols].T
    print("dat.shape:",dat.shape)
    mod=pd.DataFrame(np.array(bootstrap_data[["age","sex"]]))
    print("mod.shape:",mod.shape)

    batch = bootstrap_data['batch'].astype(int)
    batch = pd.Categorical(batch)  
    batch_col = "batch"
    covars = pd.DataFrame({batch_col: batch})
    print("covars:",covars.shape)

    output=nc.neuroCombat(dat, covars, batch_col)
    if len(sample_size)>1:
        sample_size=pd.Series(sample_size)
    return(output['data'],output['estimates']['delta.star'],output['estimates']['gamma.star'],sample_size)

def neuro_combat_bootstrap_data_sex(ids,data,n):
    #ids: batch id we are interested in 
    #data: data corresponding to ids (should only contain age, sex, batch, and features we want to harmonize)
    #n: sample size (how many samples do we want to get from one bootstrap)
    sex=data["sex"].unique()
    if len(ids)>1:#if we are interested in more than one batch
        bootstrap_data=[]
        sample_size=[]
        for ID in ids:
            size_sex={}
            for s in sex:
                data_sub = data[(data["batch"] == ID) & (data["sex"] == s)].reset_index(drop=True)            
                N=data_sub.shape[0]    
                draws=np.random.randint(0,N,size=n)#draw data with replacement
                unique_samples_size=len(np.unique(draws))
                bootstrap_data1=data_sub.iloc[draws, :].reset_index(drop=True)
                bootstrap_data.append(bootstrap_data1)
                size_sex[s]=unique_samples_size
            sample_size.append(size_sex)
            # print("id:",id)
            # print("len(draws):",len(draws))
        bootstrap_data=pd.concat(bootstrap_data)
        bootstrap_data.columns=data.columns
        print("bootstrap_data.shape:",bootstrap_data.shape)
        # sample_size=pd.DataFrame(sample_size).to_numpy()
        # print("(sample_size):",(sample_size))
    else:
        bootstrap_data=[]
        sample_size={}

        for s in sex:
            data_sub = data[(data["batch"] == ID) & (data["sex"] == s)].reset_index(drop=True)            
            N=data_sub.shape[0]    
            draws=np.random.randint(0,N,size=n)#draw data with replacement
            unique_samples_size=len(np.unique(draws))
            bootstrap_data1=data_sub.iloc[draws, :].reset_index(drop=True)
            bootstrap_data.append(bootstrap_data1)
            sample_size[s]=unique_samples_size
        bootstrap_data=pd.concat(bootstrap_data)
        bootstrap_data.columns=data.columns
        print("bootstrap_data.shape:",bootstrap_data.shape)

    #do neuro-combat for this resampled data
    feature_cols = [col for col in bootstrap_data.columns if col not in ["batch", "age", "sex"]]
    dat = bootstrap_data[feature_cols].T
    print("dat.shape:",dat.shape)
    mod=pd.DataFrame(np.array(bootstrap_data[["age","sex"]]))
    print("mod.shape:",mod.shape)

    batch = bootstrap_data['batch'].astype(int)
    batch = pd.Categorical(batch)  
    batch_col = "batch"
    covars = pd.DataFrame({batch_col: batch})
    print("covars:",covars.shape)

    output=nc.neuroCombat(dat, covars, batch_col)
    if len(sample_size)>1:
        sample_size=pd.Series(sample_size)
    return(output['data'],output['estimates']['delta.star'],output['estimates']['gamma.star'],sample_size)



def bootstrap_ntimes(ntimes,ids,data,n):
    #ntimes:how many times do we want to bootstrap
    #ids: batch id we are interested in 
    #data: data corresponding to ids (should only contain age, sex, batch, and features we want to harmonize)
    #n: sample size (how many samples do we want to get from one bootstrap)
    data_h={}
    delta={}
    gamma={}
    unique_samples={}
    for i in range(ntimes):
        data_h[i],delta[i],gamma[i],unique_samples[i]=neuro_combat_bootstrap_data(ids,data,n)
    return(data_h,delta,gamma,unique_samples)
        

def bootstrap_ntimes_sex(ntimes,ids,data,n):
    #ntimes:how many times do we want to bootstrap
    #ids: batch id we are interested in 
    #data: data corresponding to ids (should only contain age, sex, batch, and features we want to harmonize)
    #n: sample size (how many samples do we want to get from one bootstrap)
    data_h={}
    delta={}
    gamma={}
    unique_samples={}
    for i in range(ntimes):
        data_h[i],delta[i],gamma[i],unique_samples[i]=neuro_combat_bootstrap_data_sex(ids,data,n)
    return(data_h,delta,gamma,unique_samples)
        