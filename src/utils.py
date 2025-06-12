import yaml
import os
import numpy as np
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_paths(config):
    """Constructs all necessary paths from the config."""
    base_data_dir = Path(config['data_dir'])
    paths = {
        'data': base_data_dir,
        'raw_training': base_data_dir / config['raw_data_dir'] / "training",
        'raw_testing': base_data_dir / config['raw_data_dir'] / "testing",
        'processed_training': base_data_dir / config['processed_data_dir'] / "training",
        'processed_testing': base_data_dir / config['processed_data_dir'] / "testing",
        'training_params': base_data_dir / config['training_params_file'],
        'testing_params': base_data_dir / config['testing_params_file'],
        'output': Path(config['output_dir']),
    }
    return paths

def load_snapshots(path, indices, sampling_params):
    """
    Loads and samples snapshot data for a given set of simulation indices.
    
    Returns:
        A tuple of (list of snapshot matrices, list of time arrays).
    """
    X_data = []
    t_data = []
    print("Loading and sampling snapshot data...")
    for idx in indices:
        sim_path = path / f"sim_{idx}"
        X_temp = np.load(sim_path / "data.npy")
        t_temp = np.load(sim_path / "times.npy")
        
        n_steps = X_temp.shape[1]
        start_idx = int(sampling_params['start_time_percentage'] * n_steps)
        end_idx = start_idx + sampling_params['num_snapshots_fit']
        subsample = sampling_params['subsample_rate']
        
        # Ensure end_idx is within bounds
        end_idx = min(end_idx, n_steps)
        
        X_sampled = X_temp[:, start_idx:end_idx:subsample]
        t_sampled = t_temp[start_idx:end_idx:subsample]
        
        print(f"  - Sim {idx}: Loaded {t_sampled.shape[0]} snapshots.")
        
        X_data.append(X_sampled)
        t_data.append(t_sampled)

    return X_data, t_data

def get_param_cols_and_bounds(config):
    """Extracts parameter columns and bounds from config."""
    p_config = config['param_space']
    cols = p_config['param_columns']
    bounds = np.array(p_config['param_bounds'])
    if len(cols) != p_config['param_dim']:
        raise ValueError("Length of 'param_columns' must match 'param_dim'.")
    if bounds.shape[0] != p_config['param_dim']:
        raise ValueError("Length of 'param_bounds' must match 'param_dim'.")
    return cols, bounds

def normalize_params(params, bounds):
    """Normalizes parameters to the [0, 1] range."""
    p_min = bounds[:, 0]
    p_max = bounds[:, 1]
    return (params - p_min) / (p_max - p_min)

def denormalize_params(norm_params, bounds):
    """Denormalizes parameters from [0, 1] back to physical units."""
    p_min = bounds[:, 0]
    p_max = bounds[:, 1]
    return norm_params * (p_max - p_min) + p_min

def load_params(filepath, param_cols, true_val_cols=None):
    """
    Loads parameters from a text file.

    Args:
        filepath (Path): Path to the parameter file.
        param_cols (list): List of column indices for the parameters.
        true_val_cols (list, optional): List of column indices for ground truth values.

    Returns:
        A tuple containing (parameters, true_values) or just (parameters,)
    """
    data = np.loadtxt(filepath, comments="#")
    
    # Ensure param_cols is a list for proper indexing
    params = data[:, param_cols]
    
    if true_val_cols:
        true_values = data[:, true_val_cols]
        return params, true_values
    
    return params, None