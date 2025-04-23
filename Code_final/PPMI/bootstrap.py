#do bootstraps and save bootstrap outputs
"""
This script produces five outputs: n_output, d_output, bootstrap_data, n_output_b, and d_output_b.

Here, batch is another name of site.

n_output contains the results from the neural-ComBat model applied to the original 
data, using only sites with at least 6 data points.

d_output contains the results from the distributed-ComBat model applied to the 
original data, also limited to sites with at least 6 data points.

bootstrap_data is a dictionary where each key corresponds to a bootstrap iteration 
(e.g., first bootstrap, second bootstrap, etc.). Each value is a resampled dataset 
drawn from batches that contain at least 6 samples. For each batch, n samples are 
drawn with replacement, where n is the original size of that batch.

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
import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/winter_rotation/Code_final")
from helper import bootstrap_ntimes,neuro_combat_train,d_combat_train

print("===================================================================================")

common_path="/Users/xiaoqixie/Desktop/Mcgill/Winter_Rotation"
ppmi_case_folder_path=os.path.join(common_path,"PPMI")

print("import ppmi case data")
data=pd.read_csv(os.path.join(ppmi_case_folder_path,"data_with_batchID.tsv"),sep='\t')
data=data.drop(columns=["EstimatedTotalIntraCranialVol"])

print(data.columns)

#remove unused columns and rename Batch_ID to be batch
data=data.drop(columns=['file_name', 'InstitutionName', 'Manufacturer',
       'ManufacturersModelName', 'participant_id'])
#rename columns
data.rename(columns={'AGE': 'age', 'SEX': 'sex','Batch_ID':'batch'}, inplace=True)

#IDs with group size within each ID
group=data.groupby('batch').size().reset_index(name="group size")
p_size=np.unique(sorted(group['group size']))
selected_IDs=group[group['group size']>5]#at least 6 participants to ensure model working
print('minimum group size:',selected_IDs["group size"].min())

#data fit to the model
data=data[data['batch'].isin(selected_IDs["batch"])]
data.to_csv(os.path.join(ppmi_case_folder_path,"data_at_least6.csv"),index=False)
print("number of batches:",len(data['batch'].unique()))
print("========================================================================================")
file_path2=os.path.join(ppmi_case_folder_path,"combat_outputs","d_output.pkl")
file_path3=os.path.join(ppmi_case_folder_path,"combat_outputs","n_output.pkl")
#if exists, we do not otrain model again
if os.path.exists(file_path2) and os.path.exists(file_path3):
    pass
    # with open(file_path2,"rb") as f:
    #     d_output=pickle.load(f)
    # with open(file_path3,"rb") as f:
    #     n_output=pickle.load(f)
else:
    file_path=os.path.join(ppmi_case_folder_path,"combat_outputs")
    os.makedirs(file_path,exist_ok=True)
    d_output=d_combat_train(data,file_path)
    n_output=neuro_combat_train(data)

    #save outputs
    file_path1=os.path.join(ppmi_case_folder_path,"combat_outputs")
    with open(os.path.join(file_path1, "d_output.pkl"), "wb") as f:
        pickle.dump(d_output, f)
    with open(os.path.join(file_path1, "n_output.pkl"), "wb") as f:
        pickle.dump(n_output, f)

print("==================================bootstrap=======================================")
feature_name = [col for col in data.columns if col not in ["batch", "age", "sex"]]

N=1000
print(f"compute {N} times bootstrap")
bootstrap_data=bootstrap_ntimes(data,N)#****output 1

b_file=os.path.join(ppmi_case_folder_path,'bootstrap_output',f"{N}_bootstrap_data.pkl")
os.makedirs(os.path.join(ppmi_case_folder_path,'bootstrap_output'),exist_ok=True)
with open(b_file,'wb') as f:
    pickle.dump(bootstrap_data,f)

common_path1=os.path.join(ppmi_case_folder_path,"combat_outputs")

n_output_b=neuro_combat_train(bootstrap_data)#****output 2

d_file=os.path.join(common_path1,'d_combat_bootstrap_output')
os.makedirs(d_file,exist_ok=True)
d_output_b=d_combat_train(bootstrap_data,d_file)#****output 3

print("done without error!")

#save n_output_b, d_output_b as pickle file
n_file=os.path.join(common_path1,'n_combat_bootstrap_output')
os.makedirs(n_file,exist_ok=True)
with open(os.path.join(n_file,f"{N}_bootstrap_output.pkl"),'wb') as f:
    pickle.dump(n_output_b,f)

#save n_output_b, d_output_b as pickle file
d_file1=os.path.join(d_file,f"{N}_bootstrap_output.pkl")
with open(d_file1,'wb') as f:
    pickle.dump(d_output_b,f)