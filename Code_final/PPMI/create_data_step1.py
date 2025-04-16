"""
This script contains code for downloading batch effect data from the server.
It is intended to be executed on the server side.

The script downloads the data, merges all entries as rows into a single pandas DataFrame,
and saves the result as a CSV file.
"""

"""names in this directory:/data/pd/ppmi-new/bids"""
import os
import re
import shutil
import json
import pandas as pd

file_name_directory = '/data/pd/ppmi-new/bids'
file_names=os.listdir(file_name_directory)
pattern = r'^sub-\d+$'
filtered_files = [f for f in file_names if re.match(pattern, f)]

remote_files = []
for file_name in filtered_files:
    print(file_name)
    # Build the full path to the JSON files
    remote_file = os.path.join(file_name_directory, file_name, 'ses-BL', 'anat', f"{file_name}_ses-BL_acq-sag3D_run-01_T1w.json")
    remote_files.append(remote_file)
#1983 files

Data = []

# Process each remote file
for i, remote_file in enumerate(remote_files):
    try:
        with open(remote_file, 'r') as file:
            data = json.load(file)
        file_name = filtered_files[i]
        data['file_name'] = file_name
        Data.append(data)
    except FileNotFoundError:
        print(f"File not found, skipping: {remote_file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {remote_file}, skipping.")

print(len(Data)) 
local_directory = '/data/origami/xiaoqi'

Data=pd.DataFrame(Data)
Data.to_csv(os.path.join(local_directory,"Batch_effects_raw_1827.csv"),sep="\t",index=False)