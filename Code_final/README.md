## Simulated Data Scripts

The following scripts were written for simulated data. The scripts can be found in the folder `Simulation`:

### Computing Scripts
- **`simulation.py`**  
  This script is used to simulate data. The simulation method is described in detail in the report.

- **`simulation_launcher.py`**  
  This script is used to launch `simulation.py`. In this script, you need to specify several parameters to control the simulation:

  - `config`: general settings  
  - `numbers`: specifies the number of data points per site (used for equal site sizes)  
  - `size_list`: use this if you prefer unequal site sizes instead of `numbers`

  **Note**: You may want to update the `default_path` specified in both `simulation.py` and `simulation_launcher.py`.

- **`Combat_models.py`**  
  This script is used to train the Neural-Combat and D-Combat models using simulated data. It can run training multiple times across simulations.

- **`Combat_models_launcher.py`**  
  This script is used to launch `Combat_models.py`. It requires `simulation_parameters.json` as input, which is automatically saved when you run `simulation_launcher.py`.

  **Note**: You may want to update the `default_path` in both `Combat_models.py` and `Combat_models_launcher.py`.

### Visualization Scripts

These scripts are used to generate plots for simulated data for comparing harmonization results and model parameters:

- **`harmonized_data_plot.py`**  
  This script plots comparisons between:
  - Ground truth data  
  - Non-harmonized data  
  - Neural-Combat harmonized data  
  - D-Combat harmonized data  
  
  For each feature (per row), it generates four plots to visualize differences across harmonization methods.

  **Note**: You may want to update the `default_path` specified in both `simulation.py` and `simulation_launcher.py`.

- **`gamma_delta_plot.py`**  
  This script plots the gamma and delta parameters.  
  - Rows represent sites  
  - Columns represent features  
  
  Each plot includes the true gamma/delta values and the estimated values from Neural-Combat and D-Combat models.
 
  **Note**: You may want to update the `default_path` specified in both `simulation.py` and `simulation_launcher.py`.

- **`plots_launcher.py`**  
  This script is used to launch both `harmonized_data_plot.py` and `gamma_delta_plot.py`.

  **Note**: You may want to update the `default_path` specified in both `simulation.py` and `simulation_launcher.py`.

- **`sample_size.py`**  
  This script investigates whether increasing the sample size improves the performance of the neuroComBat and dComBat models.

  **Note**: Make sure to update the `default_path` to match your local directory structure.

- **`variance_gamma.py`** and **`variance_gamma.ipynb`**  
  These two versions (script and notebook) analyze the impact of gamma variance on model performance.  
  The notebook (`.ipynb`) version includes specific values and plots generated during the analysis.

#===================================================================================
## Neural-Combat and D-Combat Scripts

The following scripts are used for training the Neural-Combat and D-Combat models:

- **`distributedCombat_helpers_mod.py`**  
  This is a modified version of the original `distributedCombat_helpers.py` from GitHub. The main changes include:
  - Ensuring vectors and matrices are in the right format through computation.
  - Adding warning messages to detect and stop training if `aprior`, `bprior`, `apriorMat`, or `bpriorMat` have zero variance
  - Fixing a typo present in the original script

  The original script can be found here: [Distributed-ComBat GitHub Repository](https://github.com/andy1764/Distributed-ComBat)

- **`distributedCombat.py`**  
  This script is taken directly from the [Distributed-ComBat GitHub Repository](https://github.com/andy1764/Distributed-ComBat).

- **`neuroCombat.py`**  
  This script is taken from the [neuroCombat GitHub Repository](https://github.com/Jfortin1/neuroCombat).

- **`ComBat_train_example.py`**  
  This script provides an example of how to use the Neural-ComBat and D-ComBat models.  
  `data_example.csv` contains example data used for training two models.
  
#===================================================================================
## PPMI Data Processing Scripts

The following scripts were written for processing PPMI data which is stored on the server. The scripts can be found in the `PPMI` folder:

- **`create_data_step1.py`**  
  Loads data saved on the server and merges them into a single dataset in the format of a pandas DataFrame.

- **`create_data_step2.py`**  
  Continues the cleaning process from Step 1 by correcting institution names, identifying batch IDs (site IDs),  
  and consolidating them into a unified dataset.

- **`create_data_step3.py`**  
  Performs a sanity check on the cleaned dataset from Step 2 to identify any remaining issues such as duplicate institution names.