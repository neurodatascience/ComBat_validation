import os
import pandas as pd
import pickle
import sys
from helper import neuro_combat_train,d_combat_train

script_dir = os.path.dirname(os.path.realpath(__file__))

data=pd.read_csv(os.path.join(script_dir,"example_data.csv"))

ft_cols = [col for col in data.columns if "feature" in col]

data1=data[['age','sex','batch']+ft_cols]

output_n=neuro_combat_train(data)
output_d=d_combat_train(data,script_dir)
