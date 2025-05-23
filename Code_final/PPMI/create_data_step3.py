"""
In the script (`create_data_step2.py`), the raw data is cleaned and standardized.

This script performs a sanity check to identify and resolve any remaining duplicate institution names.
"""

import os
import pandas as pd
script_directory=os.getcwd()

# data_script=os.path.join(script_directory,'counts_sites_new.csv')
# data=pd.read_csv(data_script)

#how many sites sontain less than 10 samples
result_path="/Users/xiaoqixie/Desktop/Mcgill/winter_rotation/PPMI"
data=pd.read_csv(os.path.join(result_path,"counts_sites_new.csv"))
num=len(data['count'][data['count']<10])
print(num)
print(num/data.shape[0])

# print(data.columns)
# print(data.shape)
# print(len(data['ManufacturersModelName'].drop_duplicates()))
# #change all names to be lowercase
# data['ManufacturersModelName']=data['ManufacturersModelName'].str.lower()
# print(len(data['ManufacturersModelName'].drop_duplicates()))
# #all names are unqiue, no repeated names identified by lowercase and uppercase
# print((data['InstitutionName'].drop_duplicates()))
# #change all names to be lowercase
# data['InstitutionName']=data['InstitutionName'].str.lower()
# print((data['InstitutionName'].drop_duplicates()))
# #similar check for institution name
# print((data['Manufacturer'].drop_duplicates()))
# #change all names to be lowercase
# data['Manufacturer']=data['Manufacturer'].str.lower()
# print((data['Manufacturer'].drop_duplicates()))
# #similar check for Manufacturer
