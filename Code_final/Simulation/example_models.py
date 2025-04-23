# This script trains the Neural-Combat and D-Combat models for the report (Section 3.1).
# It generates example plots of the data before and after harmonization, 
# comparing them against the ground truth.

import os
import pandas as pd
import pickle
import sys
sys.path.append("/Users/xiaoqixie/Desktop/Mcgill/winter_rotation/Code_final")
from helper import neuro_combat_train,d_combat_train

#specify your own working directory
default_path="/Users/xiaoqixie/Desktop/Mcgill/Winter_Rotation/combat_sites"

path1="/Users/xiaoqixie/Desktop/Mcgill/winter_rotation/combat_sites/test2/N200_0.5/simulation_0"

data = pd.read_csv(os.path.join(path1, "data.csv"))
removed_columns = [col for col in data.columns if "ground_truth" in col or "epsilon" in col]
data=data.drop(columns=removed_columns)
output_n=neuro_combat_train(data)
output_d=d_combat_train(data,path1)

with open(os.path.join(path1,"output_n.pkl"),'wb') as f1:
    pickle.dump(output_n,f1)
with open(os.path.join(path1,"output_d.pkl"),'wb') as f2:
        pickle.dump(output_d,f2)
