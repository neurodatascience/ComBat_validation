## Folder Structure

### `Code_final/`
This folder contains all core scripts necessary for simulation and analysis.
- **`README.md`**: Provides detailed descriptions of each script in this directory.

#### In the `Simulation/` subfolder:
- **`simulation.py`**: Defines the simulation procedure and logic.
- **`simulation_launcher.py`**: Automates launching simulations across different settings.
---

### `combat_sites/test1`
This directory contains simulated datasets.

- Each subfolder is named by sample size, e.g., `N100`, `N200`, etc.
- Within each `N{value}` folder:
  - There are **100 independent simulations**.
  - Each simulation folder contains:
    - A `params.json` file specifying the parameters for that simulation.

---

## Simulation Parameters

- **`params.json`**: Stored inside each simulation folder, contains specific settings for that run.
- **`simulation_parameters.json`**: Global file mapping different sample sizes (`N`) to their corresponding simulation parameters.

---
