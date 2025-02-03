import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from collections import Counter
script_directory = os.getcwd()
print(script_directory)
data_directory=os.path.join(script_directory,"Batch_effects_data","Batch_effects_raw_1827.csv")
data=pd.read_csv(data_directory,sep='\t')
df=data[['InstitutionName', 'Manufacturer', 'ManufacturersModelName']]
#drop rows with NaN
df=df.dropna(axis=0)
unique_batch=df[['InstitutionName', 'Manufacturer', 'ManufacturersModelName']].drop_duplicates()
unique_batch['Batch_ID']=range(unique_batch.shape[0])
print(unique_batch)
#assign these batch_id to all rows in 
df = df.merge(unique_batch[['InstitutionName', 'Manufacturer', 'ManufacturersModelName', 'Batch_ID']], 
              on=['InstitutionName', 'Manufacturer', 'ManufacturersModelName'], 
              how='left')

# Print the updated DataFrame with the Batch_ID
print(df)
batch_ids=df['Batch_ID'].astype(int)

#try muultinomial distribution
n_trials = len(batch_ids)  
batch_id_counts = Counter(batch_ids)
batch_id_counts_df = pd.DataFrame(batch_id_counts.items(), columns=['batch_id', 'count'])
batch_id_counts_df['probability'] = batch_id_counts_df['count'] / n_trials

# print("Batch ID Probabilities:")
# print(batch_id_counts_df[['batch_id', 'probability']])

probabilities = batch_id_counts_df['probability'].values

multinomial_dist = stats.multinomial(n_trials, probabilities)

#generate random number from this distribution
random_sample = multinomial_dist.rvs(1)
print(len(random_sample[0]))
random_sample_df = pd.DataFrame(random_sample, columns=batch_id_counts_df['batch_id'])
unique_sample = random_sample_df.T.drop_duplicates().T
print(f"Unique random sample:\n{unique_sample}")