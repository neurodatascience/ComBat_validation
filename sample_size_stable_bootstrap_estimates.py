"""How many unique samples do I need for stable bootstrap estimtes?"""

print("Check mean bootstrap estimates and compare with non-bootstrap estimates")
print("Check the variance among bootstrap estimates")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import matplotlib.cm as cm
np.random.seed(666)

from sample_size_helper import neuro_combat_bootstrap_data, bootstrap_ntimes
print("===================================================================================")

common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"
ppmi_case_folder_path=os.path.join(common_path,"d-Combat_project","ppmi-age-sex-case-aseg")

print("import data showing the number of data we have for each batch id")
IDgroup=pd.read_csv(os.path.join(ppmi_case_folder_path,"batch_id_size.csv"))
top_5_ids=IDgroup.nlargest(5,"sample_size")

print("import ppmi case data")
data_80batches=pd.read_csv(os.path.join(ppmi_case_folder_path,"data_80batches.csv"))
data_80batches=data_80batches.drop(columns=["EstimatedTotalIntraCranialVol"])

data_top5=data_80batches[data_80batches["batch"].isin(top_5_ids["Batch_ID"])].reset_index(drop=True)
print(data_top5.columns)
print("data_top5.shape:",data_top5.shape)

top5IDs=data_top5["batch"].unique()
print("top5IDs:",top5IDs)
feature_name = [col for col in data_top5.columns if col not in ["batch", "age", "sex"]]
print("=============================================================================================================")
bootstrap_size=[10,20,30,40,50,60,70,80,90,100,110,120,130,140]
print("bootstrap sample size:",bootstrap_size)
print("===============================================================================================================")
print("what is the resampling sample size so that parameter estimations become stable")
print("check mean and variance")
print("=================================================================================================")
print("compute 1000 times bootstrap")
ntimes = 1000
data, delta, gamma, unique_samples = {}, {}, {}, {}

for j,n in enumerate(bootstrap_size):
    data[j],delta[j],gamma[j],unique_samples[j]=bootstrap_ntimes(ntimes,top5IDs,data_top5,n)
    print("neuro_combat done!")

b_mean={}
b_var={}
b_sizes={}
for j in range(len(bootstrap_size)):
    col_mean=[]
    col_var=[]

    unq_size=[]
    for b in range(5):#for each batches
        unique_sample_size=[]
        gamma_b=[]
        for i in range(ntimes):
            gamma_b.append(pd.Series(gamma[j][i][b,:]))
            unique_sample_size.append(unique_samples[j][i][b])
        gamma_b=pd.DataFrame.from_records(gamma_b)
        unique_sample_size=pd.DataFrame(unique_sample_size)
        unq_size.append(unique_sample_size.mean())

        col_mean.append(pd.Series(gamma_b.mean(axis=0)))
        col_var.append(pd.Series(gamma_b.var(axis=0)))

    unq_size=pd.DataFrame(unq_size)
    unq_size.index=top5IDs
    b_sizes[f"bootstrap_size{bootstrap_size[j]}"]=unq_size

    col_mean = pd.concat(col_mean,axis=1) #16x5
    print(col_mean.shape)
    col_mean.columns=top5IDs
    col_mean.index=feature_name
    col_var = pd.concat(col_var,axis=1)#16x5
    col_var.columns=top5IDs
    col_var.index=feature_name
    
    b_mean[f"bootstrap_size{bootstrap_size[j]}"]=col_mean
    b_var[f"bootstrap_size{bootstrap_size[j]}"]=col_var

print(b_mean["bootstrap_size10"].shape)
print(b_var["bootstrap_size10"].shape)
print(b_sizes["bootstrap_size10"].shape)

#look at gammas for each feature
#mean of 1000 times for each feature for different resample size
b_mean1={}
b_var1={}

for nam in feature_name:
    feature_mean=[]
    feature_var=[]
    feature_size=[]
    for nam1 in [f"bootstrap_size{n}" for n in bootstrap_size]:
        #extract feature from different sets classified by resample size
        feature_mean.append(b_mean[nam1].loc[nam])
        feature_var.append(b_var[nam1].loc[nam])
        bootstarp_mean=b_sizes[nam1].mean()
        print("bootstarp_mean:",bootstarp_mean)
        feature_size.append(bootstarp_mean)


    feature_mean=pd.concat(feature_mean,axis=1).T
    # print(feature_mean)
    feature_mean.index=[f"bootstrap_size{n}" for n in bootstrap_size]
    feature_var=pd.concat(feature_var,axis=1).T
    feature_var.index=[f"bootstrap_size{n}" for n in bootstrap_size]

    feature_size=pd.DataFrame(feature_size)
    feature_size.index=[f"bootstrap_size{n}" for n in bootstrap_size]

    print("feature_size:",feature_size)
    b_mean1[nam]=pd.concat([feature_mean,feature_size],axis=1)
    
    b_var1[nam]=pd.concat([feature_var,feature_size],axis=1)
    
    print(b_mean1[nam].shape)

# print(b_mean1)
# print(b_var1)  

b_mean_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_mean.pkl")
b_var_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_var.pkl")

# Save the dictionaries as pickle files
with open(b_mean_path, "wb") as f:
    pickle.dump(b_mean1, f)

with open(b_var_path, "wb") as f:
    pickle.dump(b_var1, f)
print("======================================================================================================")
print("for the computed 1000 times bootstraps, when we have a stable resample size?")
ntimes=1000
b_mean_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_mean.pkl")
b_var_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_var.pkl")

with open(b_mean_path, "rb") as f:
    b_mean = pickle.load(f)

with open(b_var_path, "rb") as f:
    b_var = pickle.load(f)

print("when the variance among 1000 gammas are small")
#by increasing sample size, how much improvement do we have

#for each feature, (vari-vari-1)/vari-1 percentage of inprovement
# b_var_change = {}
b_var_b_mean={}#mean of all batches
all_group={}
for feature in feature_name:
    b_var_g=b_var[feature].iloc[:, :-1]  
    all_group[feature]=b_var[feature].iloc[:, -1]#.diff().dropna() 
    #row mean
    b_var_b_mean[feature]=b_var_g.mean(axis=1)
    # var_change_df = b_var_g.pct_change().dropna()  

#    b_var_change[feature] = pd.concat([var_change_df,unique_sample],axis=1)
b_var_b_mean_df = pd.DataFrame(b_var_b_mean)#column stack
all_group_df=pd.DataFrame(all_group)
print(pd.concat([b_var_b_mean_df.mean(axis=1),all_group_df.mean(axis=1)],axis=1))








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