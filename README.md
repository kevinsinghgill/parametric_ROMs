# Parametric Reduced-Order Models

This repository contains the source code for building and evaluating parametric Reduced-Order Models (ROMs) as described in [Real-time prediction of plasma instabilities with sparse-grid-accelerated optimized dynamic mode decomposition](). The method leverages Proper Orthogonal Decomposition (POD) to find a global basis, fits local optimized Dynamic Mode Decomposition (DMD) models at various training parameter points, and uses Sparse Grid interpolation to create a continuous parametric ROM that can make predictions at new, unseen parameter points.

This implementation is designed to be general but includes specific preprocessing steps for data from gyrokinetic simulations and an analysis script for visualizing 4D phase-space data, both of which can be adapted by the user.

## Features

-   **Global Basis Creation:** Computes a low-dimensional basis from multiple high-fidelity simulations using POD (SVD).
-   **Parametric ROM Fitting:** Fits a local optimized DMD model for each training parameter point to capture its dynamics.
-   **Sparse Grid Interpolation:** Interpolates the core components of the ROMs (eigenvalues, eigenvectors, amplitudes) across the parameter space.
-   **Forecasting:** Reconstructs a ROM for any parameter point within the trained domain and generates a time-series forecast.
-   **Config-Driven Workflow:** All paths and parameters are managed through a central `config.yaml` file for easy configuration and reproducibility.

## Repository Structure

```
parametric_ROMs/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── config.yaml             # <-- Main configuration file
├── sg_lib/                 # <-- Sparse grid library (provided)
├── src/                    # <-- Core source code for the workflow
│   ├── __init__.py
│   ├── utils.py
│   ├── 0_preprocess_data.py
│   ├── 1_build_basis.py
│   ├── 2_fit_roms.py
│   ├── 3_interpolate_roms.py
│   ├── 4_forecast.py
│   └── analysis/
│       ├── __init__.py
│       └── plot_phase_space.py
└── scripts/
    └── generate_dummy_data.py  # <-- Utility to create dummy data
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/kevinsinghgill/parametric_ROMs.git
    cd parametric_ROMs
    ```

2.  Create a Python virtual environment and install the required dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## How to Run the Workflow

The entire pROM pipeline is executed by running the `src` scripts in numerical order. Before you start, you must prepare your data and configure the run.

### Step 1: Prepare Your Data

1.  **Create a `data/` directory** in the project root.
2.  Inside `data/`, create a `raw/` directory, and within that, `training/` and `testing/` subdirectories.
3.  Place your raw simulation data inside the `training` and `testing` folders, with each simulation in its own `sim_{idx}` subfolder. The final structure should look like this:
    ```
    data/
    └── raw/
        ├── training/
        │   ├── sim_1/
        │   │   └── g1.dat
        │   └── ...
        └── testing/
            └── ...
    ```
4.  Place your parameter log files (e.g., `training_parameters.txt`, `testing_parameters.txt`) directly inside the `data/` directory.

### Step 2: Configuration (`config.yaml`)

All settings are controlled by `config.yaml`. Review and edit this file before running:

-   **Paths:** Set the `output_dir` for your run.
-   **Data & Sampling:** Define the `training_indices` and `testing_indices`. Adjust snapshot sampling as needed.
-   **Parameter Space:** This section is crucial for mapping your physical simulation parameters to the normalized `[0, 1]` space required for interpolation.
    -   `param_dim`: The number of physical parameters you are varying.
    -   `param_columns`: The column indices from your parameter log files that correspond to these dimensions.
    -   `param_bounds`: A list of `[min, max]` values for each parameter. **You must set these bounds to correctly encompass all your training and testing data.**
-   **`testing_setup`**: This block controls how test points are generated.
    -   `mode: 'generate'`: For exact reproducibility. Ignores the physical parameter values in your test log file and generates random points in `[0, 1]` using a fixed seed.
    -   `mode: 'load'`: Loads physical parameters directly from your test log file and normalizes them.

**Example `testing_parameters.txt` format:**
The last two columns are assumed to be ground-truth values for validation.
```
# idx  tau      q0       ...  growth_rate  freq
1      1.34493  5.43775  ...  102.218      -425.998
2      1.64473  3.75583  ...  143.815      -184.055
...
```

**WARNING:** The sparse grid interpolation script (`3_interpolate_roms.py`) assumes that the order of your training simulations **exactly matches** the order of the sparse grid points generated by the `sg_lib` library for the given level. This is a strong requirement for the current implementation.

### Step 3: Run the Pipeline

Execute the scripts sequentially from the root directory.

```bash
# (Optional) Run the dummy data generator to test the pipeline setup
python scripts/generate_dummy_data.py

# 0. Preprocess raw data into a standard .npy format
python src/0_preprocess_data.py --config config.yaml

# 1. Build the global POD basis from all training data
python src/1_build_basis.py --config config.yaml

# 2. Fit a local DMD model at each training parameter point
python src/2_fit_roms.py --config config.yaml

# 3. Interpolate the ROM components using a sparse grid
python src/3_interpolate_roms.py --config config.yaml

# 4. Use the interpolated model to forecast for the test parameters
python src/4_forecast.py --config config.yaml
```
After completion, all outputs—including the global basis, trained ROM components, a validation report, and final forecasts—will be saved in the directory specified by `output_dir`.

### Step 4: Analysis and Visualization

After running the workflow, you can generate plots to compare the ROM's prediction with the ground truth data.

```bash
# Generate a plot for the test case specified in config.yaml
python src/analysis/plot_phase_space.py --config config.yaml

# Or, override the config and specify a different test index
python src/analysis/plot_phase_space.py --config config.yaml --test_idx 5
```
Plots will be saved in the `outputs/your_run_name/plots/` directory.

## Customizing for Your Data

The script `src/0_preprocess_data.py` is specific to a binary data format. To use this repository with your own data, you will need to:
1.  Modify the `read_gyrokinetics_binary` function inside `src/0_preprocess_data.py` to read your file format.
2.  Ensure your function returns two NumPy arrays: a `data_matrix` of shape `(N, T)` where `N` is the flattened state dimension and `T` is the number of time steps, and a `times` array of shape `(T,)`.
3.  The `src/analysis/plot_phase_space.py` script is specific to 4D data. You will need to adapt the data loading and reshaping portion for your problem's dimensionality.
