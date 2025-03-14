import pandas as pd
import os


print("Import data")
common_path="/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation"

#os.path.realpath(os.path.dirname(__file__))

data_file="ppmi-age-sex-case-aseg"

Data_path=os.path.join(common_path,"d-ComBat_project",data_file)

file_name=f'data_with_batchID'
Data=pd.read_csv(os.path.join(Data_path,f'{file_name}.tsv'),sep="\t")

print("========================================================================")
print("sample size of batch ID group")
group = Data.groupby("Batch_ID").size().reset_index(name="sample_size")
print("the number of groups:",group.shape)
#we have 123 groups
print("include batch ID if the group size is at least 2")#at least two samples are needed for combat and bootstrap
group_filtered=group[group["sample_size"]>=2]
print("the number of filtered groups:",group_filtered.shape)
#we have 80 groups
print("===========================================================================")
Data_subset = Data[Data["Batch_ID"].isin(group_filtered["Batch_ID"])]
print(Data.shape,Data_subset.shape)#from 1342 to 1299
print("===============================================================================")

def drop_rename(data):
    data=Data_subset.drop(columns=["file_name","InstitutionName",
                            "Manufacturer","ManufacturersModelName",
                            "participant_id"])  
    data=data.rename(columns={"AGE":"age","SEX":"sex","Batch_ID":"batch"})     
    return(data)


top_10_batches = group.nlargest(10, "sample_size")
Data_sub=Data[Data["Batch_ID"].isin(top_10_batches["Batch_ID"])]
Data_sub=drop_rename(Data_sub)
Data_sub.to_csv(os.path.join(Data_path,"data_top10batches.csv"),index=False)           