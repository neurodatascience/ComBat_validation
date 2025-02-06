"""In the script Batch_analysis.py, I cleaned raw data and corrected institution name if they are from the same institution but names are different.
I also removed some records if the insitution is not specified correctly, such as names like 'Section of BrainImaging','.'.
After cleaness, we have 1714 records left and we have 131 sites for counting batch effects (unqiue combinations of InstitutionName, Manufacturer, ManufacturersModelName).
The number of sites may change in the future if mistakes are found during the process of correcting insitution names.
"""

import os
import pandas as pd
script_directory=os.getcwd()
"""Check if we still have duplicates of names"""
data_script=os.path.join(script_directory,'counts_sites_new.csv')
data=pd.read_csv(data_script)
print(data.columns)
print(data.shape)
print(len(data['ManufacturersModelName'].drop_duplicates()))
#change all names to be lowercase
data['ManufacturersModelName']=data['ManufacturersModelName'].str.lower()
print(len(data['ManufacturersModelName'].drop_duplicates()))
#all names are unqiue, no repeated names identified by lowercase and uppercase
print((data['InstitutionName'].drop_duplicates()))
#change all names to be lowercase
data['InstitutionName']=data['InstitutionName'].str.lower()
print((data['InstitutionName'].drop_duplicates()))
#similar check for institution name
print((data['Manufacturer'].drop_duplicates()))
#change all names to be lowercase
data['Manufacturer']=data['Manufacturer'].str.lower()
print((data['Manufacturer'].drop_duplicates()))
#similar check for Manufacturer

#number of samples per site suggested by ComBat and d-ComBat papers
"""The original ComBat paper metioned that when the batch size (the number of batches) is less than 10, empirical bayesian will not work well.
The federated averging paper further metioned the effetiveness of their method for unblanaced,dependent data. The federated averaging algorithm is tested on data with small atch size and moderate numbers within each batch )sample size per batch is at least number of batches).
d-ComBat paper uses 53 batches, with a total of 505 samples aross all sites (approximately 10 samples per site). Siements 213, Philips 70, GE 222.

Thus, I will start from 131 sites with 20 samples per site. 
"""

data_script=os.path.join(script_directory,'batch_data_cleaned.csv')
data=pd.read_csv(data_script)
print(data.columns)
print(data.shape)
