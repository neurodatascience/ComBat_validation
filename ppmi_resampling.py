import numpy as np
import pandas as pd
import scipy.stats as stats
import os

script_dir=os.path.realpath(os.path.dirname(__file__))
# print("script_dir",script_dir)
ppmi=pd.read_csv(os.path.join(script_dir,"batch_data_ppmi.tsv"),sep='\t')

#site size of ppmi 
sample_size = ppmi.groupby("Batch_ID").size().reset_index(name="number")
sample_size = sample_size[sample_size["number"] >= 10]#only consider site with at least 10 samples

df = ppmi[['Batch_ID', 'InstitutionName', 'Manufacturer', 'ManufacturersModelName']].drop_duplicates()
df1 = df[df["Batch_ID"].isin(sample_size["Batch_ID"])]

#subset based on sample size
data=ppmi.drop(columns=['InstitutionName', 'Manufacturer', 'ManufacturersModelName','file_name','participant_id','EstimatedTotalIntraCranialVol'])
data1=data[data["Batch_ID"].isin(df1["Batch_ID"])]
# print(data1)
data1 = data1.rename(columns={
    "AGE": "age",
    "SEX": "sex",
    "Batch_ID": "batch"
})#rename columns
data1.to_csv(os.path.join(script_dir,"resampling_data","data.csv"),index=False)

grouped_dict = {batch_id: group.reset_index(drop=True) for batch_id, group in data1.groupby("batch")}

data2 = ppmi.drop(columns=[
    'InstitutionName', 'Manufacturer', 'ManufacturersModelName',
    'file_name', 'participant_id', 'EstimatedTotalIntraCranialVol',
    'AGE', 'SEX', 'Batch_ID'
])

np.random.seed(666)
Data = []
for i in grouped_dict.keys():
    d = grouped_dict[i].copy()
    n = 4*d.shape[0]
    
    # Bootstrap sampling with replacement
    draws = np.random.randint(0, n/4, size=n)
    new_d = d.iloc[draws, :].reset_index(drop=True)
        
    """what if noise is bigger (enlarge site effect by artificial noise)"""
    # Generate gaussian noise for each feature
    #add noise centered at group mean
    d1=d[data2.columns]

    a=d1.mean(axis=0)   
    b=d1.std(axis=0)#np.random.randint(1,10,size=len(data2.columns))
    print(len(a),len(b),len(data2.columns))
    noise = [stats.norm.rvs(a[z], b[z], size=n) for z in range(len(data2.columns))]  

    # Apply noise to numeric columns only
    j = 0
    for column in data2.columns:
        if np.issubdtype(new_d[column].dtype, np.number):  # Check if numeric
            new_d[column] = new_d[column] + noise[j]
        j += 1
    # a1=d["age"].mean()
    # b1=d["age"].std()
    # noise1=stats.norm.rvs(a1, b1, size=n)
    # new_d['age']=new_d['age']+noise1
    new_d["batch"] = i
    Data.append(new_d)

# Concatenate DataFrames
Data = pd.concat(Data, ignore_index=True)
# Save output
file_name="data_with_noise_4n_1"
Data.to_csv(os.path.join(script_dir,"resampling_data" ,f'{file_name}.csv'), index=False)
print("saved")