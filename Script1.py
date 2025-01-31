"""Testing the distribution of features"""
import os
import pandas as pd
import numpy as np
from spicy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# %%

# %%
data_path="/Users/xiaoqixie/Desktop/Winter_Rotation/d-ComBat_project/qpn-age-sex-hc-aseg.tsv"
data=pd.read_csv(data_path,sep='\t')
data=pd.DataFrame(data)

# %%
df = data.drop(['participant_id','EstimatedTotalIntraCranialVol','SEX'], axis=1)

# %%
print(df.apply(min))
print(df.apply(max))


# %%
def fit_normal(column):
    mu, sigma = stats.norm.fit(column)
    return pd.Series({'mu': mu, 'sigma': sigma})
def fit_exponential(column):
    loc, scale = stats.expon.fit(column, floc=0)  #the distribution starts from 0
    return pd.Series({'loc': loc, 'scale': scale})
def fit_t(column):
    df_t,loc_t,scale_t=stats.t.fit(column)
    return pd.Series({'df_t':df_t,'loc_t':loc_t,'scale_t':scale_t})
def fit_gamma(column):
    column = pd.to_numeric(column, errors='coerce').dropna()
    if column.empty:
        return pd.Series({'shape_g': np.nan, 'loc_g': np.nan, 'scale_g': np.nan})
    shape_g, loc_g, scale_g = stats.gamma.fit(column)
    return pd.Series({'shape_g': shape_g, 'loc_g': loc_g, 'scale_g': scale_g})
def fit_log_normal(column):
    column = pd.to_numeric(column, errors='coerce').dropna()
    if column.empty:
        return pd.Series({'shape_ln': np.nan, 'loc_ln': np.nan, 'scale_ln': np.nan})
    shape_ln, loc_ln, scale_ln = stats.lognorm.fit(column)
    return pd.Series({'shape_ln': shape_ln, 'loc_ln': loc_ln, 'scale_ln': scale_ln})



# %%
fit_results_n = df.apply(fit_normal)
fit_results_e=df.apply(fit_exponential)
fit_results_t=df.apply(fit_t)
fit_results_gamma=df.apply(fit_gamma)
fit_results_log_normal=df.apply(fit_log_normal)


# %%

# %%
#generate random numbers from distributions for each feature and lchoose the one matching sample distribution best.

# %%
#fit_results_e each column is the st of parameters of a feature similar for fit_results_n: 2x18

# %%
def generate_normal_samples(params,num_samples=500):
    return np.random.normal(loc=params['mu'],scale=params['sigma'],size=num_samples)
def generate_exponential_samples(params, num_samples=500):
    return np.random.exponential(scale=params['scale'], size=num_samples)


# %%
save_root = '/Users/xiaoqixie/Desktop/Winter_Rotation/d-ComBat_project'
for i in range(df.shape[1]):
    print(f"Processing column {i + 1}/{df.shape[1]}: {df.columns[i]}")
    
    data1 = df.iloc[:, i].dropna()  # Drop NaN values
    column_name = df.columns[i]

    # Check if the column is empty after cleaning
    if data1.empty:
        print(f"Skipping {column_name} (Empty or Non-Numeric Data)")
        continue

    # Generate x values for smooth plotting
    x = np.linspace(data1.min(), data1.max(), 100)

    # Fit and Compute PDFs for Different Distributions 
    
    # Normal Distribution
    params_n = fit_results_n.iloc[:, i]
    mu, sigma = params_n['mu'], params_n['sigma']
    pdf_normal = stats.norm.pdf(x, mu, sigma) if not np.isnan(mu) and not np.isnan(sigma) else None

    # Exponential Distribution
    params_e = fit_results_e.iloc[:, i]
    loc, scale = params_e['loc'], params_e['scale']
    pdf_exponential = stats.expon.pdf(x, loc, scale) if not np.isnan(loc) and not np.isnan(scale) else None

    # Student's t-Distribution
    params_t = fit_results_t.iloc[:, i]
    df_t, loc_t, scale_t = params_t['df_t'], params_t['loc_t'], params_t['scale_t']
    pdf_t = stats.t.pdf(x, df_t, loc_t, scale_t) if not np.isnan(df_t) and not np.isnan(loc_t) and not np.isnan(scale_t) else None

    # Gamma Distribution
    params_g = fit_results_gamma.iloc[:, i]
    shape_g, loc_g, scale_g = params_g['shape_g'], params_g['loc_g'], params_g['scale_g']
    pdf_gamma = stats.gamma.pdf(x, shape_g, loc_g, scale_g) if not np.isnan(shape_g) and not np.isnan(loc_g) and not np.isnan(scale_g) else None

    # Log-normal Distribution
    params_ln = fit_results_log_normal.iloc[:, i]
    shape_ln, loc_ln, scale_ln = params_ln['shape_ln'], params_ln['loc_ln'], params_ln['scale_ln']
    pdf_lognorm = stats.lognorm.pdf(x, shape_ln, loc_ln, scale_ln) if not np.isnan(shape_ln) and not np.isnan(loc_ln) and not np.isnan(scale_ln) else None

    # --- Plot the Histogram and Distributions ---
    plt.figure(figsize=(12, 8))
    plt.hist(data1, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black', label='Sample Distribution')

    if pdf_normal is not None:
        plt.plot(x, pdf_normal, 'r-', label=f"Normal ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$)")
    
    if pdf_exponential is not None:
        plt.plot(x, pdf_exponential, 'g-', label=f"Exponential (loc={loc:.2f}, scale={scale:.2f})")

    if pdf_t is not None:
        plt.plot(x, pdf_t, 'purple', linestyle='--', label=f"Student's t (df={df_t:.2f})")

    if pdf_gamma is not None:
        plt.plot(x, pdf_gamma, 'orange', linestyle='-.', label=f"Gamma (shape={shape_g:.2f})")

    if pdf_lognorm is not None:
        plt.plot(x, pdf_lognorm, 'brown', linestyle='dotted', label=f"Log-normal (shape={shape_ln:.2f})")

    # Labels and Title
    plt.xlabel('Data Values')
    plt.ylabel('Density')
    plt.title(f'{column_name}: Sample vs Fitted Distributions')
    plt.legend()

    # Save the plot
    save_path = os.path.join(save_root, f'feature_{column_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    # Show the plot
    plt.show()

# %%
#let's further do qq plot
for i in range(df.shape[1]):
    print(f"Processing column {i + 1}/{df.shape[1]}: {df.columns[i]}")

    data1 = df.iloc[:, i].dropna() 
    column_name = df.columns[i]

    # Generate x values for smooth plotting
    x = np.linspace(data1.min(), data1.max(), 100)

    # Normal Distribution
    params_n = fit_results_n.iloc[:, i].to_dict()
    mu, sigma = params_n.get('mu', np.nan), params_n.get('sigma', np.nan)

    # Exponential Distribution
    params_e = fit_results_e.iloc[:, i].to_dict()
    loc, scale = params_e.get('loc', np.nan), params_e.get('scale', np.nan)

    # Student's t-Distribution
    params_t = fit_results_t.iloc[:, i].to_dict()
    df_t, loc_t, scale_t = params_t.get('df_t', np.nan), params_t.get('loc_t', np.nan), params_t.get('scale_t', np.nan)

    # Gamma Distribution
    params_g = fit_results_gamma.iloc[:, i].to_dict()
    shape_g, loc_g, scale_g = params_g.get('shape_g', np.nan), params_g.get('loc_g', np.nan), params_g.get('scale_g', np.nan)

    # Log-normal Distribution
    params_ln = fit_results_log_normal.iloc[:, i].to_dict()
    shape_ln, loc_ln, scale_ln = params_ln.get('shape_ln', np.nan), params_ln.get('loc_ln', np.nan), params_ln.get('scale_ln', np.nan)

    # --- Subplots for Q-Q Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))  # 2 rows, 3 columns
    axes = axes.flatten()  # Flatten to index subplots easier

    # Normal Q-Q Plot
    if np.isfinite(mu) and np.isfinite(sigma):
        stats.probplot(data1, dist="norm", plot=axes[0])
        axes[0].set_title(f'{column_name}: Normal Q-Q Plot')

    # Exponential Q-Q Plot
    if np.isfinite(loc) and np.isfinite(scale):
        stats.probplot(data1, dist="expon", plot=axes[1])
        axes[1].set_title(f'{column_name}: Exponential Q-Q Plot')

    # Student's t Q-Q Plot
    if np.isfinite(df_t) and np.isfinite(loc_t) and np.isfinite(scale_t):
        stats.probplot(data1, dist=stats.t, sparams=(df_t, loc_t, scale_t), plot=axes[2])
        axes[2].set_title(f'{column_name}: Student\'s t Q-Q Plot')

    # Gamma Q-Q Plot
    if np.isfinite(shape_g) and np.isfinite(loc_g) and np.isfinite(scale_g):
        stats.probplot(data1, dist="gamma", sparams=(shape_g, loc_g, scale_g), plot=axes[3])
        axes[3].set_title(f'{column_name}: Gamma Q-Q Plot')

    # Log-normal Q-Q Plot
    if np.isfinite(shape_ln):
        stats.probplot(data1, dist="lognorm", sparams=(shape_ln,loc_ln, scale_ln), plot=axes[4])
        axes[4].set_title(f'{column_name}: Log-normal Q-Q Plot')

    # Hide any unused subplots
    for j in range(5, 6):  
        fig.delaxes(axes[j])  # Remove the last empty subplot

    # Improve layout
    plt.tight_layout()
    
    # Save the full figure
    save_path = os.path.join(save_root, f'qq_plots_{column_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# %%
# Extract the column correctly
sex = data['SEX']

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

x = np.array([0, 1])  # Possible outcomes (0 or 1)
pmf_bernoulli = stats.binom.pmf(x, n, p)

# Print PMF values
print(f"P(X=0) = {pmf_binomial[0]:.4f}")
print(f"P(X=1) = {pmf_binomial[1]:.4f}")

# Print estimated parameters
print(f"Estimated Binomial Parameters: n={n}, p={p:.4f}")

