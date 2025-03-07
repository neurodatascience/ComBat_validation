import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pickle
script_dir=os.path.realpath(os.path.dirname(__file__))
print("import data")
file_name="data_with_noise_4n_1"
data=pd.read_csv(os.path.join(script_dir,"resampling_data",f"{file_name}.csv"))
print(data.columns)
def plot_pca(data, batch_labels, title):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=batch_labels, palette="tab10", alpha=1)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

Data=data.drop(columns=['age','sex','batch'])
batch_id=data['batch']
# plot_pca(Data, batch_id, "PCA of Non-Harmonized Data")
neuro_combat=pd.read_csv(f"/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{file_name}/neuro_data.csv")
neuro_combat=neuro_combat.T
# plot_pca(neuro_combat, batch_id, "PCA of Harmonized Data")

file_path=f'/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites/{file_name}'
filenames =os.listdir(file_path)
filenames=[f for f in filenames if "site_out" in f]
sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(sorted_filenames)

sites=[]
for filename in sorted_filenames:
    file_full_path = os.path.join(file_path, filename) 

    with open(file_full_path, "rb") as f:
        site_data = pickle.load(f)
        sites.append(site_data["dat_combat"])
d_combat = np.column_stack(sites).T
print(d_combat.shape)

def plot_pca_subplots(data1, batch_labels1, title1, data2, title2, data3, title3, save_path=None):
    
    pca1 = PCA(n_components=2)
    pca_result1 = pca1.fit_transform(data1)

    pca2 = PCA(n_components=2)
    pca_result2 = pca2.fit_transform(data2)

    pca3 = PCA(n_components=2)
    pca_result3 = pca3.fit_transform(data3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)  # Create 3 subplots

    # Find consistent axis limits across all PCA plots
    x_min = min(pca_result1[:, 0].min(), pca_result2[:, 0].min(), pca_result3[:, 0].min())
    x_max = max(pca_result1[:, 0].max(), pca_result2[:, 0].max(), pca_result3[:, 0].max())
    y_min = min(pca_result1[:, 1].min(), pca_result2[:, 1].min(), pca_result3[:, 1].min())
    y_max = max(pca_result1[:, 1].max(), pca_result2[:, 1].max(), pca_result3[:, 1].max())

    # Plot first PCA (Non-Harmonized)
    scatter1 = sns.scatterplot(x=pca_result1[:, 0], y=pca_result1[:, 1], hue=batch_labels1, palette="tab10", alpha=0.8, ax=axes[0], legend=False)
    axes[0].set_title(title1)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # Plot second PCA (Neuro-ComBat)
    scatter2 = sns.scatterplot(x=pca_result2[:, 0], y=pca_result2[:, 1], hue=batch_labels1, palette="tab10", alpha=0.8, ax=axes[1], legend=False)
    axes[1].set_title(title2)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    # Plot third PCA (D-ComBat)
    scatter3 = sns.scatterplot(x=pca_result3[:, 0], y=pca_result3[:, 1], hue=batch_labels1, palette="tab10", alpha=0.8, ax=axes[2], legend=True)
    axes[2].set_title(title3)
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)


    handles, labels = scatter3.get_legend_handles_labels()
    if len(labels) > 0:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=min(len(labels), 3), title="Batch Labels")

    # plt.tight_layout(rect=[0, 0, 1, 1.05])  

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved at: {save_path}")

    plt.show()

save_path=os.path.join("/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/combat_sites",
                             file_name,"PCA.png")
# feature_name=Data.columns
neuro_combat = pd.DataFrame(neuro_combat)
neuro_combat.columns = Data.columns

d_combat = pd.DataFrame(d_combat)
d_combat.columns = Data.columns

print(Data.columns)

col = Data.columns
#['Left-Lateral-Ventricle', 'Left-Thalamus']
#only look at batch 15 and 26
# s=np.where(batch_id.isin([15,26]))[0]
# print(s)
# Data1=Data.iloc[s,:]
# neuro_combat1=neuro_combat.iloc[s,]
# d_combat1=d_combat.iloc[s,]
# batch_id1=batch_id.iloc[s,]
plot_pca_subplots(
    Data, batch_id, "PCA of Non-Harmonized Data for feature", 
    neuro_combat, "PCA of neuro-combat Harmonized Data for feature",
    d_combat, "PCA of d-combat Harmonized Data for feature", 
    save_path 
)