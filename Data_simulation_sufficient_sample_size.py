"""
This script calculates the RMSE of harmonized data (gamma and delta) compared to the ground truth, 
using estimators from neuro-ComBat. 
Estimates from d-ComBat are not shown, as their outputs are nearly identical.
"""
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,48,60,72,84,90,100,120,140,160,180,200,220,240,260,280,300]


default_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites"
output_path={}
file="Homogeneous_Heterogenity_Heterogenity"
for x in X:
    output_path[x]=os.path.join(default_path,f"min_points{x}",
                            f"{file}_nonlinear_N{x*5}_G100_I5_Gamma4")
os.makedirs(os.path.join(default_path,file),exist_ok=True)
data={}
ground_truth={}
for x in X:
    data[x]=pd.read_csv(os.path.join(output_path[x],
                                     f"{file}_nonlinear_N{x*5}_G100_I5_Gamma4.csv"))
    ground_truth[x]=data[x][[col for col in data[x].columns if "ground" in col]].T

output={}
for x in X:
    with open(os.path.join(output_path[x],"output_n.pkl"),"rb") as f:
        output[x]=pickle.load(f)

def rowwise_RMSE(mat1, mat2):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return np.sqrt(np.mean((mat1 - mat2) ** 2, axis=1))

#from output to combat data
ncombat={}
for x in X:
    ncombat[x]=pd.DataFrame(output[x]["combat_data"])

rmse_results = {}
for x in X:
    r=[]
    rmse_results[x] = rowwise_RMSE(ncombat[x], ground_truth[x])

rmse_results_avg=[]
for x in X:
    rmse_results_avg.append(rmse_results[x].mean())
    print("harmonized data rmse for features")
    print(f"{x}:",rmse_results[x])
#rmse not showing that by adding data, the harmonization data is closer to the ground truth
plt.figure(figsize=(20, 8))
plt.plot(X,pd.Series(rmse_results_avg), marker='o')
plt.xlabel("sample size")
plt.ylabel("rmse of harmonized y")
plt.xticks(X, rotation=90)
plt.savefig(os.path.join(default_path,file,"rmse_sample_size.png"))
plt.close()
#let's look at gamma and delta

gamma={}
gamma_n={}

delta={}
delta_n={}
rmse_gamma={}
rmse_delta={}
rmse_gamma_avg=[]
rmse_delta_avg=[]
for x in X:
    gamma[x]=pd.read_csv(os.path.join(output_path[x],"gamma_IG.csv"))#3x5
    gamma_n[x]=output[x]["gamma_star"].T
    print(gamma_n[x])
    rmse_gamma[x]=rowwise_RMSE(gamma[x],gamma_n[x])
    rmse_gamma_avg.append(rmse_gamma[x].mean())
    print("gamma rmse over features:")
    print(f"{x}:",rmse_gamma[x].mean())

plt.figure(figsize=(20, 8))
plt.plot(X,pd.Series(rmse_gamma_avg), marker='o')
plt.xlabel("sample size")
plt.ylabel("rmse of gamma")
plt.xticks(X, rotation=90)
plt.savefig(os.path.join(default_path,file,"rmse_gamma.png"))
plt.close()

for x in X:
    delta[x]=pd.read_csv(os.path.join(output_path[x],"delta_IG.csv"))#3x5
    delta_n[x]=output[x]["delta_star"].T
    rmse_delta[x]=rowwise_RMSE(delta[x],delta_n[x])
    rmse_delta_avg.append(rmse_delta[x].mean())
    print("delta rmse over features:")
    print(f"{x}:",rmse_delta[x].mean())

plt.figure(figsize=(20, 8))
plt.plot(X,pd.Series(rmse_delta_avg), marker='o') 
plt.xlabel("sample size")
plt.ylabel("rmse of delta")
plt.xticks(X, rotation=90)
plt.savefig(os.path.join(default_path,file,"rmse_delta.png"))
plt.close()

"""
Homogeneous data, homo sex and homo age:
Conclusion: As we gradually increase the sample size of the simulated data, 
the RMSE of the harmonized data becomes more stable. 
Specifically, when the sample size exceeds 20, RMSE fluctuations decrease noticeably, with 24 appearing to be a turning point. 

Homogeneous data, inhomo sex and inhomo age:
Conclusion: As we gradually increase the sample size of the simulated data, 
the RMSE of the harmonized data becomes more stable. 
Specifically, when the sample size exceeds 20, RMSE fluctuations decrease noticeably, with 26 appearing to be a turning point. 

"""
