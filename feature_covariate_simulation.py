import pandas as pd
import numpy as np
from scipy import stats

def generate_fc_data(data_path, num_samples, num_sites):
    """Generate dataset based on fitted distributions from real data."""
    
    data = pd.read_csv(data_path, sep='\t')#real data is a tsv file
    data = pd.DataFrame(data)

    np.random.seed(666)

    # fit distributions
    def fit_normal(column):
        mu, sigma = stats.norm.fit(column)
        return pd.Series({'mu': mu, 'sigma': sigma})

    def fit_log_normal(column):
        column = pd.to_numeric(column, errors='coerce').dropna()
        if column.empty:
            return pd.Series({'shape_ln': np.nan, 'loc_ln': np.nan, 'scale_ln': np.nan})
        shape_ln, loc_ln, scale_ln = stats.lognorm.fit(column)
        return pd.Series({'shape_ln': shape_ln, 'loc_ln': loc_ln, 'scale_ln': scale_ln})
    def fit_bernoulli(column):
        """Fit a bernoulli distribution to a binary categorical variable."""
        unique_values = column.dropna().unique()
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(f"Column {column.name} is not binary (0/1). Found values: {unique_values}")

        n = 1 
        p = column.mean()  #prob of having 1

        return pd.Series({'n': n, 'p': p})

    # fit distributions
    sex = data['SEX'].astype(int)
    fit_b = fit_bernoulli(sex)
    n, p = fit_b['n'], fit_b['p']

    age = data['AGE'].astype(int)
    fit_results_n = fit_normal(age)
    mu, sigma = fit_results_n['mu'], fit_results_n['sigma']

    df = data[['Left-Caudate', 'Left-Lateral-Ventricle', 'Left-Putamen', 
               'Right-Lateral-Ventricle', 'Right-Thalamus']]
    fit_results_log_normal = df.apply(fit_log_normal)

    df1 = data.drop(columns=['participant_id', 'EstimatedTotalIntraCranialVol', 'AGE', 'SEX', 
                            'Left-Caudate', 'Left-Lateral-Ventricle', 'Left-Putamen', 
                            'Right-Lateral-Ventricle', 'Right-Thalamus'])
    fit_results_n = df1.apply(fit_normal)

    Data1 = []
    
    for j in range(num_sites):
        sex_r = np.random.binomial(n, p, num_samples)
        age_r = np.random.normal(mu, sigma, num_samples)

        random_samples_df = pd.DataFrame()
        for i in range(df.shape[1]):
            params_ln = fit_results_log_normal.iloc[:, i]
            shape_ln, loc_ln, scale_ln = params_ln['shape_ln'], params_ln['loc_ln'], params_ln['scale_ln']
            random_values = np.random.lognormal(mean=np.log(scale_ln), sigma=shape_ln, size=num_samples)
            random_samples_df[df.columns[i]] = random_values
        random_samples_df1 = pd.DataFrame()
        for i in range(df1.shape[1]):
            params_n = fit_results_n.iloc[:, i]
            mu, sigma = params_n['mu'], params_n['sigma']
            random_values = np.random.normal(mu, sigma, num_samples)
            random_samples_df1[df1.columns[i]] = random_values        
        Data = pd.DataFrame()
        Data['age'] = age_r
        Data['sex'] = sex_r
        Data[random_samples_df.columns] = random_samples_df
        Data[random_samples_df1.columns] = random_samples_df1
        Data['site']=j
        Data1.append(Data)
    # Data1=pd.concat(Data1)
    return Data1  # Returns a list of DataFrames

# Allow running standalone
if __name__ == "__main__":
    data_path = "/Users/xiaoqixie/Desktop/Mcgill/Rotations/Winter_Rotation/d-ComBat_project/qpn-age-sex-hc-aseg.tsv"
    generated_data = generate_fc_data(data_path,20,131)
    print(f"Generated {len(generated_data)} datasets.")
    # print(generated_data[0])

