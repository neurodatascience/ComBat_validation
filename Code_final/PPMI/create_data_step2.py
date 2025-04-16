"""
This script continues processing the data saved in 'create_data_step1.py'.

Key steps include:
- Correcting variations in 'InstitutionName' when they refer to the same institution
- Removing records with improperly specified institution names (e.g., 'Section of BrainImaging', '.', etc.)

After cleaning:
- 1,714 valid records remain
- 131 unique sites are identified for batch effect analysis, based on the combination of:
  'InstitutionName', 'Manufacturer', and 'ManufacturersModelName'

Note: The number of sites may change in the future if additional inconsistencies in institution names are discovered.

### Outputs:
1. 'batch_data_cleaned.csv':  
   Contains the cleaned data with standardized institution names and a new column, 'Batch_ID',  
   which represents a unique combination of 'InstitutionName', 'Manufacturer', and 'ManufacturersModelName'.

2. 'counts_sites_new.csv':  
   A summary file showing the number of data points for each Batch_ID.
"""


import os
import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats
# import numpy as np
from collections import Counter

script_directory = os.getcwd()
print(script_directory)
data_directory=os.path.join(script_directory,"Batch_effects_data","Batch_effects_raw_1827.csv")
data=pd.read_csv(data_directory,sep='\t')
df=data[['InstitutionName','Manufacturer','ManufacturersModelName']]
data1 = df.copy() 
data1 = data1.dropna(axis=0)

"""Set names for the same institution to be same"""

data1.loc[:, 'InstitutionName'] = data1['InstitutionName'].str.strip('"')#remove double quotes
data1.loc[:, 'InstitutionName'] = data1['InstitutionName'].str.strip('')#remove single quote

# Perform replacements
institution_replacements = {
    #Remove these rows directly
    'Section of BrainImaging': None,  
    'Imaging Center':None,
    'IMPORT_INST':None,
    '.':None,
    ################################################
    'WHITNEY IMAGING CENTER': 'Whitney Imaging Center',
    'WHITNEY': 'Whitney Imaging Center',
    'Whitney Imaging': 'Whitney Imaging Center',
    #
    'Radiology at La Jolla': 'UCSD Radiology at La Jolla',
    #
    'Spruce MRI Assoc': 'Pennsylvania Hospital Spruce',
    #
    'PPMI': 'PPMI_2-0_Imaging_Core',
    #
    'NORTHWESTERN MEMORIAL HOSPITAL/43e63f': 'NORTHWESTERN MEMORIAL HOSPITAL',
    #
    'JHH MRI': 'Johns Hopkins Hospital',
    'JHH MR01 TRIO': 'Johns Hopkins Hospital',
    'JHH': 'Johns Hopkins Hospital',
    #
    'JHU MR01 Trio':'Johns Hopkins University',
    'JHU MR01':'Johns Hopkins University',
    'JHU MR01':'Johns Hopkins University',
    'JHU MR01 Trio':'Johns Hopkins University',
    'JHU MR01 TRIO':'Johns Hopkins University',
    'JHU TRIO MR01':'Johns Hopkins University',
    #
    'HOSPITAL CLINIC BARCELONA': 'Hospital Clinico de Barcelona',
    #
    'HARBORVIEW PRISMA 3T x42460': 'Harborview Medical Center',
    'HARBORVIEW MEDICAL CENTER TRIO': 'Harborview Medical Center',
    'Harborview Med Ctr GE x42460': 'Harborview Medical Center',
    'Harborview Med Ctr 3T x42460':'Harborview Medical Center',
    'HMC':'Harborview Medical Center',
    #
    'Emory University .F75457.': 'Emory University',
    'Emory University .A324DA.': 'Emory University',
    #
    'El Camino Hospital MV': 'El Camino Hospital',
    #
    'Diagnostic Centers of America Boca': 'Diagnostic Centers of America',
    'DIAGNOSTIC CENTERS OF AMERICA': 'Diagnostic Centers of America',
    'Diag_Ctrs_America_Boca': 'Diagnostic Centers of America',
    #
    'CHARING CROSS HOSPITAL': 'Charing Cross Hospital',
    #
    'CCHS Cleveland Clinc Mellen Trio': 'CCHS Cleveland Clinc',
    #
    'BOSTON MEDICAL CENTER HAC': 'Boston Medical Center',
    'BMC': 'Boston Medical Center',
    #
    'Baylor College Medicine': 'Baylor College of Medicine',
    #
    'Uni-Tuebingen':'U of Tuebingen',
    ' Uni-Tuebingen':'U of Tuebingen',
    #
    'Univ of South Florida':'USF',
    #
    'CCHS CLEVELAND CLINIC':'CCHS Cleveland Clinic',
    'CCHS Cleveland Clinc':'CCHS Cleveland Clinic',
    #
    'Neurologie T?bingen':'Neurologie Uni Tuebingen',
    'Neurologie':'Neurologie Uni Tuebingen',
    #
    'Neuroradiologie':'Neuroradiologie Uni Tuebingen',
    'Neuroradiologie Uni Tuebingen':'Neuroradiologie Uni Tuebingen',
    #
    'UAB - Cardiovascular MRI':'UAB',
    'UAB- Cardiovascular MRI':'UAB',
    #
    'USF':'UCSF Medical Center',
    #
    'Anon':'Anonymous Hospital',
}

for old_name, new_name in institution_replacements.items():
    if new_name is None:  # If None, remove rows with that institution name
        data1 = data1.loc[data1['InstitutionName'] != old_name, :]
    else:
        data1.loc[data1['InstitutionName'] == old_name, 'InstitutionName'] = new_name
unique_batch=data1[['InstitutionName', 'Manufacturer', 'ManufacturersModelName']].drop_duplicates()
unique_batch['Batch_ID']=range(unique_batch.shape[0])
print(unique_batch['Batch_ID'])
#assign these batch_id to all rows in 
data2 = data1.merge(unique_batch[['InstitutionName', 'Manufacturer', 'ManufacturersModelName', 'Batch_ID']], 
              on=['InstitutionName', 'Manufacturer', 'ManufacturersModelName'], 
              how='left')

#assign these batch_id to all rows in 
data2 = data1.merge(unique_batch[['InstitutionName', 'Manufacturer', 'ManufacturersModelName', 'Batch_ID']], 
              on=['InstitutionName', 'Manufacturer', 'ManufacturersModelName'], 
              how='left')

columns_to_lower = ['InstitutionName', 'Manufacturer', 'ManufacturersModelName']
data2[columns_to_lower] = data2[columns_to_lower].apply(lambda x: x.str.lower())
print(data2)
data2.to_csv(os.path.join(script_directory,'batch_data_cleaned.csv'),index=False)

batch_ids=data2['Batch_ID'].astype(int)
batch_id_counts = Counter(batch_ids)
batch_id_counts_df = pd.DataFrame(batch_id_counts.items(), columns=['Batch_ID', 'count'])

aggregated_data = data2.groupby('Batch_ID', as_index=False).agg({
    'InstitutionName': 'first',  # Taking the first occurrence of each Batch_ID
    'Manufacturer': 'first',     # Same for other fields
    'ManufacturersModelName': 'first'
})

data3 = batch_id_counts_df.merge(aggregated_data, on='Batch_ID', how='left')
print(data3)
loc_directory="/Users/xiaoqixie/Desktop/Winter_Rotation/d-ComBat_project"
data3.to_csv(os.path.join(loc_directory,"counts_sites_new.scv"),index=False)
