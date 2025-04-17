"""
This script explores the appropriate sample size when working with simulated data.
It begins by using RMSE as the metric for evaluating model performance, 
but later transitions to cosine similarity. 
The rationale is that preserving the overall shape or pattern after harmonization is more important than minimizing pointwise differences in Euclidean space.
"""
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/winter_rotation/Code_final")
from helper import ci_plot,harmonized_plot
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

default_path="/Users/xiaoqixie/Desktop/Mcgill/winter_rotation/combat_sites"

default_path=os.path.join(default_path,"test1")

with open(os.path.join(default_path,"simulation_parameters.json"),"r") as f:
    simulation_parameters=json.load(f)#contains all simulation_parameters used in simulation.py

# For each sample size, I simulated 100 datasets.
# I calculated the RMSE for each simulation, then identified the 2.5th and 97.5th percentiles of the RMSE values (sorted in ascending order).
# Finally, I plotted the 95% confidence interval along with the mean RMSE across the 100 simulations.
##****#
F=[]
lower=0.025
upper=0.975

RMSE = {}
Bounds = {}
RMSE_avg = {}
MinMax = {}
Variance = {}

for i, config in enumerate(simulation_parameters):
    N = config["N"]
    simulation_times = config["simulation_times"]
    F.append(N)

    RMSE[i] = []
    sample_dir = os.path.join(default_path, f"N{N}")

    for s in range(simulation_times):
        print(f"Sample size index {i}, simulation {s}")
        sim_path = os.path.join(sample_dir, f"simulation_{s}")


        data = pd.read_csv(os.path.join(sim_path, 'data.csv'))
        gt_cols = [col for col in data.columns if "ground_truth" in col]
        ground_truth = data[gt_cols].T
        print(ground_truth.shape)

        with open(os.path.join(sim_path, 'output_n.pkl'), 'rb') as f:
            output = pickle.load(f)
        combat_data = output['combat_data']
        print(combat_data.shape)

        rmse = np.sqrt(np.mean((ground_truth - combat_data) ** 2))
        RMSE[i].append(rmse)

    RMSE[i] = np.array(RMSE[i])

    RMSE_avg[i] = np.mean(RMSE[i])
    Bounds[i] = np.quantile(RMSE[i], [lower, upper])
    MinMax[i] = [np.min(RMSE[i]), np.max(RMSE[i])]
    Variance[i] = np.var(RMSE[i])

F=np.array(F)
RMSE_avg = np.array([RMSE_avg[i] for i in range(len(simulation_parameters))])
Bounds = np.stack([Bounds[i] for i in range(len(simulation_parameters))], axis=1)
MinMax = np.stack([MinMax[i] for i in range(len(simulation_parameters))], axis=1)
Variance=np.array([Variance[i] for i in range(len(simulation_parameters))])
print(Bounds.shape)

df = pd.DataFrame({
    'Sample Size': F,
    'RMSE': RMSE_avg,
    'Lower Bound': Bounds[0, :],
    'Upper Bound': Bounds[1, :],
    'Minimal': MinMax[0, :],
    'Maximal': MinMax[1, :],
    'Variance': Variance
}).reset_index(drop=True)

df.to_csv(os.path.join(default_path,"rsme_table.csv"),index=False)

ci_plot(df,"RMSE of Harmonized y",default_path,95,"RMSE")
############################################################################################
# Conclusion:
# RMSE does not clearly indicate that increasing the sample size leads to more stable estimations.
# However, when the variance of additive site effects is small, the harmonized data tends to be closer to the ground truth.
#############################################################################################
# In real-world data, the ground truth is unknown. 
# Therefore, preserving the overall shape or pattern after harmonization is more important 
# than matching the exact values at each point (i.e., minimizing pointwise differences in Euclidean space).
#estimate cosine similarity

#compute cosine-similarity for features for all simulated data
F=[]
lower=0.025
upper=0.975
CS={}#cosine-similarity
Bounds={}
CS_mean={}
Bounds1={}

scaler = StandardScaler()

for i, config in enumerate(simulation_parameters):
    print(i)
    F.append(config["N"])
    N=config["N"]
    simulation_times=config["simulation_times"]
    CS[i]=[]
    for s in range(simulation_times):
        path=os.path.join(default_path,f"N{N}",f"simulation_{s}")
        data=pd.read_csv(os.path.join(path,'data.csv'))
        gt_cols=[col for col in data.columns if "ground_truth" in col]
        ground_truth=data[gt_cols].T

        with open(os.path.join(path,'output_n.pkl'),'rb') as f:
            output=pickle.load(f)
        combat_data=output['combat_data']
        #CS
        value_list=[]
        for row in range(len(gt_cols)):
            similarity = cosine_similarity(ground_truth.iloc[row, :].values.reshape(1, -1),
                  combat_data[row, :].reshape(1, -1))[0, 0]
            value_list.append(similarity)
        value_list=np.array(value_list)
        CS[i].append(np.mean(value_list))
    CS[i]=np.array(CS[i])
    CS_mean[i]=np.mean(CS[i])
    Bounds[i]=np.quantile(CS[i], [lower, upper])
    Bounds1[i]=[np.min(CS[i]), np.max(CS[i])]
    
F=np.array(F)
CS_mean = np.array([CS_mean[i] for i in range(len(simulation_parameters))])
Bounds = np.stack([Bounds[i] for i in range(len(simulation_parameters))], axis=1)
Bounds1 = np.stack([Bounds1[i] for i in range(len(simulation_parameters))], axis=1)

df = pd.DataFrame({
    'Sample Size': F,
    'Cosine Similarity mean': CS_mean,
    'Lower Bound': Bounds[0, :],
    'Upper Bound': Bounds[1, :],
    'Minimal': Bounds1[0, :],
    'Maximal': Bounds1[1, :]
}).reset_index(drop=True)

df.to_csv(os.path.join(default_path,"cosine_similarity_table.csv"),index=False)

ci_plot(df,"Cosine Similarity of Harmonized y",default_path,95,'Cosine Similarity mean')
#Conclusion:
#cosine-similarity also does not show that the increase of sample size can make a pattern closer to the groudn truth. 

###############################################################################################