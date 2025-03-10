"""In tihs script, I run two models neuro-combat and d-combat"""
import pandas as pd
import numpy as np
import distributedCombat as dc
import os
import pickle
import neuroCombat as nc

print("Import data")
common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"

data_file="ppmi-age-sex-case-aseg"

Data_path=os.path.join(common_path,"d-ComBat_project",data_file)

file_name=f'data_80batches'
Data=pd.read_csv(os.path.join(Data_path,f'{file_name}.csv'))

def dat_function(Data,data_type):
    if data_type=="simulated":
        dat_columns=[name for name in Data.columns if "feature" in name]
        dat=pd.DataFrame(np.array(Data[dat_columns]).T)
    elif data_type=="not_simulated":
        dat=pd.DataFrame(np.array(Data.drop(columns=["batch","age","sex"]).T))
    print("dat.shape:",dat.shape)
    return(dat)

data_type="not_simulated"
dat=dat_function(Data,data_type)
mod=pd.DataFrame(np.array(Data[["age","sex"]]))
print("mod.shape:",mod.shape)
batch = Data['batch'].astype(int)
batch = pd.Categorical(batch)  
batch_col = "batch"
covars = pd.DataFrame({batch_col: batch})
print("covars:",covars.shape)

os.makedirs(os.path.join(common_path,'combat_sites',data_file),exist_ok=True)

print("dat.shape",dat.shape)
com_out = nc.neuroCombat(dat, covars, batch_col)
# print(com_out['data'])
d=pd.DataFrame(com_out['data'])
d.to_csv(os.path.join(common_path,f"combat_sites/{data_file}/neuro_data.csv"),index=False)

file_path = os.path.join(common_path, f'combat_sites/{data_file}/neuro_combat.pickle')
with open(file_path, "wb") as f:
    pickle.dump(com_out, f)

print("===============================================================================")
### Step 1
print("Step 1")
site_outs = []
for b in covars[batch_col].unique():
    s = covars[batch_col] == b  
    df = dat.loc[:, s.to_numpy()]  
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = f"{common_path}/combat_sites/{data_file}/site_out_" + str(b) + ".pickle"
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
    f = f"{common_path}/combat_sites/{data_file}/site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(df, bat, x, verbose=True, central_out=central, file=f)
    site_outs.append(f)

central = dc.distributedCombat_central(site_outs)

# print("central['var_pooled']:", central['var_pooled'])

def RMSE(v1,v2):
    return(np.sqrt(((v1-v2)**2).mean()))
print("===========================================================================")

### Compare distributed vs original
print("Step 3")

if data_type=="simulated":
    site_outs = []
    error = []
    perror = []  # percent difference
    rmse=[]
    for b in covars[batch_col].unique():
        s = list(map(lambda x: x == b, covars[batch_col]))
        df = dat.loc[:, s]
        bat = covars[batch_col][s]
        x = mod.loc[s, :]
        f = f"{common_path}/combat_sites/{data_file}/site_out_" + str(b) + ".pickle"
        out = dc.distributedCombat_site(df, bat, x, central_out=central, file=f)
        site_outs.append(f)
        """compare with grouth truth"""
        data1=Data.iloc[s,:]
        col_names = [name for name in Data.columns if "feature" in name]
        G=len(col_names)
        data2=data1[col_names]
        v1=com_out["data"][:, s] - out["dat_combat"]
        # print('out["dat_combat"]:',out["dat_combat"].shape)
        v2=abs(com_out["data"][:, s] - out["dat_combat"]) / com_out["data"][:, s]
        rmse_c=[]
        rmse_d=[]
        for q in range(G):
            v3=data2.iloc[:,q]
            v4=com_out["data"][q, s]
            rmse_c.append(RMSE(v3,v4).round(5))
            v5=out["dat_combat"].iloc[q,:]
            rmse_d.append(RMSE(v3,v5).round(5))
        d1=pd.DataFrame({f"rmse_combat{b}":rmse_c,f"rmse_d_comba{b}t":rmse_d})
        rmse.append(d1)

        print(f'error in batch{b}:',v1.min(axis=1),v1.max(axis=1))
        print(f'perror in batch{b}:',v2.min(axis=1),v2.max(axis=1))
        
        e=com_out["data"][:, s] - out["dat_combat"]
        error.append(e)
        pe=abs(com_out["data"][:, s] - out["dat_combat"]) / com_out["data"][:, s]
        perror.append(pe)
elif data_type=="not_simulated":
    site_outs = []
    for b in covars[batch_col].unique():
        s = list(map(lambda x: x == b, covars[batch_col]))
        df = dat.loc[:, s]
        bat = covars[batch_col][s]
        x = mod.loc[s, :]
        f = f"{common_path}/combat_sites/{data_file}/site_out_" + str(b) + ".pickle"
        out = dc.distributedCombat_site(df, bat, x, central_out=central, file=f)
        site_outs.append(f)
    
print("central['var_pooled']:", central['var_pooled'])

if data_type=="simulated":
    rmse=pd.concat(rmse, axis=1)
    print("RMSE:",rmse)
    rmse.to_csv(os.path.join(common_path,f"combat_sites/{data_file}/rmse.csv"),index=False)
    error=pd.concat(error)
    perror=pd.concat(perror)

    print("max error", (error).max(axis=1).max())
    print("max perror", (perror).max(axis=1).max())

    print("min error", (error).min(axis=1).min())
    print("min perror", (perror).min(axis=1).min())

print("===============================================================")

