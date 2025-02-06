"""In tihs script, I test if two models ComBat and d-ComBat work using data simulated from real data, where the simulation is done by calling generate_fc_data function."""

from feature_covariate_simulation import generate_fc_data
import pandas as pd
import numpy as np
# from rpy2.robjects import r, numpy2ri, pandas2ri
import os
import sys
sys.path.append('/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation')
import distributedCombat as dc
script_directory=os.getcwd()
data_path = "/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/d-ComBat_project/qpn-age-sex-hc-aseg.tsv"
num_samples=20
num_sites=131
data=generate_fc_data(data_path,num_samples,num_sites)
print(len(data))

#d-ComBat
mod=[]#number of sample per site x covaiates
dat=[]#features x number of sample per site
for i in range(len(data)):
    d=data[i]
    m=d[['age','sex']]
    dd=d.drop(columns=['age','sex','site']).T
    mod.append(m)
    dat.append(dd)
    print(dd.empty,m.empty)

"""let my bat have the same class as in author's code"""
batch1 = np.tile(np.arange(1, 5), 100 // 4) 
batch1 = pd.Categorical(batch1)  
batch_col = "batch"
covars = pd.DataFrame({batch_col: batch1})
""""""
test_sites_path = os.path.join(script_directory, 'test_sites')
os.makedirs(test_sites_path, exist_ok=True)
no reference batch
site_outs = []    
for i in range(num_sites):
    d=data[i]
    bat = pd.Series(d['site'], dtype=covars[batch_col].dtype)
    df=dat[i]
    x=mod[i]
    f = os.path.join(test_sites_path,"site_out_" + str(i) + ".pickle")
    out = dc.distributedCombat_site(df, bat, x, verbose=True, file=f)
    site_outs.append(f)

central = dc.distributedCombat_central(site_outs)
print(central.keys())
B_hat=pd.DataFrame(central['B_hat'])
print(B_hat.shape)

#ComBat
import neuroCombat as nc
data1=pd.concat(data)
# print(data1.shape)
# print(len(dat))
dat1=data1.drop(columns=['age','sex','site']).T
# print(dat1.shape)
mod1=data1[['age','sex']]
# print(mod1.shape)
batch2 = data1['site'].astype(int) 
batch2 = pd.Categorical(batch2)  
batch_col = "batch"
covars1 = pd.DataFrame({batch_col: batch2})
# print(len(covars1))
com_out = nc.neuroCombat(dat1, covars1, batch_col)#no reference batch
#com_out_ref = nc.neuroCombat(dat, covars, batch_col, ref_batch="1")
print(com_out)