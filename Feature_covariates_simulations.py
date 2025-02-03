import os
import pandas as pd
import numpy as np
from spicy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

data_path="/Users/xiaoqixie/Desktop/Winter_Rotation/d-ComBat_project/qpn-age-sex-hc-aseg.tsv"
data=pd.read_csv(data_path,sep='\t')
data=pd.DataFrame(data)

#global set up
num_samples = 100
def fit_normal(column):
    mu, sigma = stats.norm.fit(column)
    return pd.Series({'mu': mu, 'sigma': sigma})

def fit_log_normal(column):
    column = pd.to_numeric(column, errors='coerce').dropna()
    if column.empty:
        return pd.Series({'shape_ln': np.nan, 'loc_ln': np.nan, 'scale_ln': np.nan})
    shape_ln, loc_ln, scale_ln = stats.lognorm.fit(column)
    return pd.Series({'shape_ln': shape_ln, 'loc_ln': loc_ln, 'scale_ln': scale_ln})

#for covraite sex, we fit a bernoulli distribution
sex=data['SEX'].astype(int)
# Fit binomial distribution
def fit_bernoulli(column):
    """Fit a bernoulli distribution to a binary categorical variable."""
    # Ensure data is binary (0/1)
    unique_values = column.dropna().unique()
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"Column {column.name} is not binary (0/1). Found values: {unique_values}")

    n = 1 
    p = column.mean()  #prob of 1

    return pd.Series({'n': n, 'p': p})

fit_b = fit_bernoulli(sex)
n, p = fit_b['n'], fit_b['p']

#generate random numbers
  # Number of random samples you want to generate
sex_r = np.random.binomial(n, p, num_samples)
print(sex_r)

#for covariates age, we fit a normal distribution
age=data['AGE'].astype(int)
fit_results_n = fit_normal(age)
mu, sigma = fit_results_n['mu'], fit_results_n['sigma']

age_r = np.random.normal(mu, sigma, num_samples)
print(age_r)

#for features Left-Caudate,Left-Lateral-Ventricle,Left-Putamen,Right-Lateral-Ventricle,Right-Thalamus (log_normal)
df=data[['Left-Caudate','Left-Lateral-Ventricle','Left-Putamen','Right-Lateral-Ventricle','Right-Thalamus']]
fit_results_log_normal=df.apply(fit_log_normal)
#generate random samples
random_samples_df = pd.DataFrame()
for i in range(df.shape[1]):
    params_ln = fit_results_log_normal.iloc[:, i]
    shape_ln, loc_ln, scale_ln = params_ln['shape_ln'], params_ln['loc_ln'], params_ln['scale_ln']
    random_values = np.random.lognormal(mean=np.log(scale_ln), sigma=shape_ln, size=num_samples)
    random_samples_df[df.columns[i]] = random_values

print(random_samples_df)

#for the rest of featres, we fit normal dstribution.
df=data.drop(columns=['participant_id','EstimatedTotalIntraCranialVol','AGE','SEX','Left-Caudate', 'Left-Lateral-Ventricle', 'Left-Putamen', 'Right-Lateral-Ventricle', 'Right-Thalamus'])
print(df.columns)
fit_results_n = df.apply(fit_normal)
random_samples_df1 = pd.DataFrame()
for i in range(df.shape[1]):
    params_n = fit_results_n.iloc[:, i]
    mu, sigma = params_n['mu'], params_n['sigma']
    random_values = np.random.normal(mu, sigma, num_samples)
    random_samples_df1[df.columns[i]] = random_values
print(random_samples_df1) 

#combine together to be a dataset
Data=pd.DataFrame()
Data['age']=age_r
Data['sex']=sex_r
Data[random_samples_df.columns]=random_samples_df
Data[random_samples_df1.columns]=random_samples_df1
print(Data)

#do check by plotting histograms of simulated data
for i in range(Data.shape[1]):
    # Extract each column
    df = Data.iloc[:, i]  
    plt.figure()  
    plt.hist(df, bins=10, edgecolor='black')  
    plt.title(f"Histogram of {Data.columns[i]}") 
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()  

    #'Left-Caudate', 'Left-Lateral-Ventricle', 'Left-Putamen', 'Right-Lateral-Ventricle', 'Right-Thalamus' log normal