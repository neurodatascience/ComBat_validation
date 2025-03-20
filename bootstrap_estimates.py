"""How many unique samples do I need for stable bootstrap estimtes?"""

print("Check mean and confidence interval of bootstrap estimates and compare with non-bootstrap estimates")
print("Check the variance among bootstrap estimates")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import matplotlib.cm as cm
np.random.seed(666)

from bootstrap_helper import bootstrap_ntimes,neuro_combat_train,d_combat_train
print("===================================================================================")

common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"
ppmi_case_folder_path=os.path.join(common_path,"d-Combat_project","ppmi-age-sex-case-aseg")

print("import ppmi case data")
data_80batches=pd.read_csv(os.path.join(ppmi_case_folder_path,"data_80batches.csv"))
data_80batches=data_80batches.drop(columns=["EstimatedTotalIntraCranialVol"])

#IDs with group size within each ID
group=data_80batches.groupby('batch').size().reset_index(name="group size")
p_size=np.unique(sorted(group['group size']))

selected_IDs=group[group['group size']>6]#at least 6 participants to ensure model working
print('minimum group size:',selected_IDs["group size"].min())

data=data_80batches[data_80batches['batch'].isin(selected_IDs["batch"])]
print("========================================================================================")
output=neuro_combat_train(data)
print("======================================================================================")
file_path=os.path.join(ppmi_case_folder_path,'d_combat_bootstrap_output')
output=d_combat_train(data,file_path)
feature_name = [col for col in data_80batches.columns if col not in ["batch", "age", "sex"]]
print("=============================================================================================================")
print("compute 1000 times bootstrap")
N=1000
bootstrap_data=bootstrap_ntimes(data,N)
# print(bootstrap_data)
# n_ouput=neuro_combat_train(bootstrap_data)
d_output=d_combat_train(bootstrap_data,file_path)
print("done without error!")


"""ComBat could even insert biases when data is too heterogeneous."""
#alpha, beta is unqiue for each feature. 
# gamma,delta  for each feature from different batches follows same distribution.
#
# n_output=neuro_combat_train(bootstrap_data)




# #two plots, one for gamma and one for delta and colors marking values from different resample sets

# colors = cm.get_cmap("tab10", len(bootstrap_size))  # Using 'tab10' with enough colors

# feature_name=data_80batches.drop(columns=["batch","age","sex","EstimatedTotalIntraCranialVol"]).columns
# for i in range(len(top5IDs)):  # for each id
#     plt.figure(figsize=(18, 14))  

#     for j in range(len(bootstrap_size)):
#         gamma = results[j]["gamma"][i, :]  
#         plt.plot(feature_name, gamma, color=colors(j), label=f"bootstrap_size {bootstrap_size[j]}")

#     plt.xlabel("Resampling size")
#     plt.ylabel("Gamma estimation")

#     plt.xticks(rotation=45, ha='right')  

#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 

    
#     save_path=os.path.join(ppmi_case_folder_path,"resampled sample plot")
#     os.makedirs(save_path,exist_ok=True)
#     plt.savefig(os.path.join(save_path,f"gamma_id{top5IDs[i]}.png"))

#     plt.close()