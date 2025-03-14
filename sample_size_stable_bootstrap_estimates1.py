"""How many unique samples do I need for stable bootstrap estimtes?"""

print("Check mean bootstrap estimates and compare with non-bootstrap estimates")
print("Check the variance among bootstrap estimates")

import os
import pandas as pd
import numpy as np
import pickle
from sample_size_helper import bootstrap_ntimes_sex

np.random.seed(666)
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
bootstrap_size=[10,20,30,40,50,60,70,80,90,100]#sample size per sex
print("bootstrap sample size:",bootstrap_size)
print("===============================================================================================================")
print("what is the resampling sample size so that parameter estimations become stable")
print("check mean and variance")
print("=================================================================================================")
print("compute 1000 times bootstrap")
ntimes = 1000
data, delta, gamma, unique_samples = {}, {}, {}, {}

for j,n in enumerate(bootstrap_size):
    data[j],delta[j],gamma[j],unique_samples[j]=bootstrap_ntimes_sex(ntimes,top5IDs,data_top5,n)
    print("neuro_combat done!")

b_mean={}
b_var={}
b_sizes={}
for j in range(len(bootstrap_size)):
    col_mean=[]
    col_var=[]

    unq_size=[]
    for b in range(len(top5IDs)):#for each batches
        unique_sample_size=[]
        gamma_b=[]
        for i in range(ntimes):
            gamma_b.append(pd.Series(gamma[j][i][b,:]))
            unique_sample_size.append(unique_samples[j][i][b])
        gamma_b=pd.DataFrame.from_records(gamma_b)
        unique_sample_size=pd.DataFrame(unique_sample_size)
        # print(unique_sample_size.shape)
        unq_size.append(unique_sample_size.mean(axis=0))

        col_mean.append(pd.Series(gamma_b.mean(axis=0)))
        col_var.append(pd.Series(gamma_b.var(axis=0)))

    unq_size=pd.DataFrame(unq_size)
    unq_size.index=top5IDs
    unq_size.columns=[0,1]

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
        bootstrap_mean=b_sizes[nam1].mean(axis=0)
        print("bootstarp_mean:",bootstrap_mean)
        feature_size.append(bootstrap_mean)


    feature_mean=pd.concat(feature_mean,axis=1).T
    # print(feature_mean)
    feature_mean.index=[f"bootstrap_size{n}" for n in bootstrap_size]
    feature_var=pd.concat(feature_var,axis=1).T
    feature_var.index=[f"bootstrap_size{n}" for n in bootstrap_size]

    feature_size=pd.concat(feature_size,axis=1).T
    feature_size.index=[f"bootstrap_size{n}" for n in bootstrap_size]

    # print("feature_size:",feature_size)
    b_mean1[nam]=pd.concat([feature_mean,feature_size],axis=1)
    
    b_var1[nam]=pd.concat([feature_var,feature_size],axis=1)
    
    print(b_mean1[nam].shape)

# print(b_mean1)
# print(b_var1)  

b_mean_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_mean_sex.pkl")
b_var_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_var_sex.pkl")

# Save the dictionaries as pickle files
with open(b_mean_path, "wb") as f:
    pickle.dump(b_mean1, f)

with open(b_var_path, "wb") as f:
    pickle.dump(b_var1, f)
# print("======================================================================================================")
print("for the computed 1000 times bootstraps, when we have a stable resample size?")
ntimes=1000
b_mean_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_mean_sex.pkl")
b_var_path = os.path.join(ppmi_case_folder_path, f"bootstrap{ntimes}_var_sex.pkl")

with open(b_mean_path, "rb") as f:
    b_mean = pickle.load(f)

with open(b_var_path, "rb") as f:
    b_var = pickle.load(f)

print("when the variance among 1000 gammas are small")
#by increasing sample size, how much improvement do we have

#for each feature, (vari-vari-1)/vari-1 percentage of inprovement
# b_var_change = {}
b_var_b_mean={}#mean of all batches
all_0={}
all_1={}
for feature in feature_name:
    b_var_g=b_var[feature].iloc[:, :-2]  
    all_0[feature]=b_var[feature].iloc[:, -2]#.diff().dropna()
    all_1[feature]=b_var[feature].iloc[:, -1]
    # print(all_group[feature]) 
    #row mean
    b_var_b_mean[feature]=b_var_g.mean(axis=1)
    # var_change_df = b_var_g.pct_change().dropna()  

#    b_var_change[feature] = pd.concat([var_change_df,unique_sample],axis=1)
b_var_b_mean_df = pd.DataFrame(b_var_b_mean)#column stack
all_0_df=pd.DataFrame(all_0)
all_1_df=pd.DataFrame(all_1)
print(pd.concat([b_var_b_mean_df.mean(axis=1),all_0_df.mean(axis=1),all_1_df.mean(axis=1),],axis=1))
