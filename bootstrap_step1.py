#do bootstraps and save bootstrap outputs
"""This script will have three outputs: bootstrap_data, n_output_b, d_output_b.

bootstrap_data is a dictionary where each key corresponds to a bootstrap iteration 
(e.g., the first bootstrap, second bootstrap, etc.). 
Each value is a resampled dataset drawn from batches that have at least 6 samples.
For each batch, n samples were drawn with replacement, where n is the original size of that batch.


n_output_b contains the neuro-combat model results.
n_output_b is a dictionary where each key corresponds to a bootstrap iteration 
(e.g., the first bootstrap, second bootstrap, etc.). 
Each value under a key is itself a dictionary with the following entries:
  - 'combat_data': the harmonized data.
  - 'alpha': the standardized mean.
  - 'beta': the coefficients for the fixed effects ('sex' and 'age').
  - 'XB': the product of the fixed effects and their corresponding coefficients.
  - 'delta_star': the estimated delta parameter.
  - 'gamma_star': the estimated gamma parameter.

d_output_b contains the d-combat model results.
d_output_b is a dictionary where each key corresponds to a bootstrap iteration 
(e.g., the first bootstrap, second bootstrap, etc.). 
Each value under a key is itself a dictionary with a key name showing batch ID,
with the following entries:
  - 'combat_data': the harmonized data.
  - 'alpha': the standardized mean.
  - 'beta': the coefficients for the fixed effects ('sex' and 'age').
  - 'XB': the product of the fixed effects and their corresponding coefficients.
  - 'delta_star': the estimated delta parameter.
  - 'gamma_star': the estimated gamma parameter.
  

"""


import os
import pandas as pd
import numpy as np
import pickle

np.random.seed(666)

from bootstrap_helper import bootstrap_ntimes,neuro_combat_train,d_combat_train

print("===================================================================================")

common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"
ppmi_case_folder_path=os.path.join(common_path,"d-Combat_project","ppmi-age-sex-case-aseg")
num_batches=20

print("import ppmi case data")
data_80batches=pd.read_csv(os.path.join(ppmi_case_folder_path,"data_80batches.csv"))
data_80batches=data_80batches.drop(columns=["EstimatedTotalIntraCranialVol"])

#IDs with group size within each ID
group=data_80batches.groupby('batch').size().reset_index(name="group size")
p_size=np.unique(sorted(group['group size']))

selected_IDs=group.sort_values(by="group size", ascending=False).head(num_batches)
#group[group['group size']>5]#at least 6 participants to ensure model working
print('minimum group size:',selected_IDs["group size"].min())

#data fit to the model
data=data_80batches[data_80batches['batch'].isin(selected_IDs["batch"])]
print("number of batches:",len(data['batch'].unique()))
print("========================================================================================")
file_path=os.path.join(ppmi_case_folder_path,'d_combat_bootstrap_output',f'{num_batches}batches')
d_output=d_combat_train(data,file_path)
n_output=neuro_combat_train(data)
print("======================================================================================")
feature_name = [col for col in data_80batches.columns if col not in ["batch", "age", "sex"]]

N=1000
print(f"compute {N} times bootstrap")
bootstrap_data=bootstrap_ntimes(data,N)#****output 1

b_file=os.path.join(ppmi_case_folder_path,'bootstrap_output',f'{num_batches}batches',f"{N}_bootstrap_data.pkl")
os.makedirs(os.path.join(ppmi_case_folder_path,'bootstrap_output',f'{num_batches}batches'),exist_ok=True)
with open(b_file,'wb') as f:
    pickle.dump(bootstrap_data,f)

n_output_b=neuro_combat_train(bootstrap_data)#****output 2

file_path1=os.path.join(file_path,f"{N}bootstraps")
d_output_b=d_combat_train(bootstrap_data,file_path1)#****output 3

print("done without error!")

#save n_output_b, d_output_b as pickle file
n_file=os.path.join(ppmi_case_folder_path,'n_combat_bootstrap_output',f'{num_batches}batches',f"{N}_bootstrap_output.pkl")
os.makedirs(os.path.join(ppmi_case_folder_path,'n_combat_bootstrap_output',f'{num_batches}batches'),exist_ok=True)
with open(n_file,'wb') as f:
    pickle.dump(n_output_b,f)

#save n_output_b, d_output_b as pickle file
d_file=os.path.join(file_path,f"{N}_bootstrap_output.pkl")
with open(d_file,'wb') as f:
    pickle.dump(d_output_b,f)
