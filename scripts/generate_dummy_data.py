import numpy as np
import os
import yaml
from pathlib import Path

def generate_dummy_files(sim_dir, state_dim, n_times):
    """Creates dummy data.npy and times.npy files."""
    os.makedirs(sim_dir, exist_ok=True)
    # Generate some simple sine waves as dummy data
    t = np.linspace(0, 100, n_times)
    freq1 = 2 * np.pi * 0.1
    freq2 = 2 * np.pi * 0.5
    data = (np.sin(freq1 * t) * np.random.randn(state_dim, 1) + 
            np.cos(freq2 * t) * np.random.randn(state_dim, 1))
    
    np.save(sim_dir / "data.npy", data.astype(np.complex128))
    np.save(sim_dir / "times.npy", t.astype(np.float64))
    print(f"  - Created dummy data in {sim_dir}")

def main():
    print("Generating dummy data based on config.yaml...")
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    base_data_dir = Path(config['data_dir'])
    proc_dir = base_data_dir / config['processed_data_dir']
    
    state_dim = 1000 # A reasonable dummy state dimension
    n_times = 5000   # Dummy number of time steps

    # Generate processed training data
    train_dir = proc_dir / "training"
    for idx in config['data_params']['training_indices']:
        generate_dummy_files(train_dir / f"sim_{idx}", state_dim, n_times)

    # Generate processed testing data
    test_dir = proc_dir / "testing"
    for idx in config['data_params']['testing_indices']:
        generate_dummy_files(test_dir / f"sim_{idx}", state_dim, n_times)

    # Generate dummy parameter files
    dim = config['sg_interpolation']['param_dim']
    n_train = len(config['data_params']['training_indices'])
    n_test = len(config['data_params']['testing_indices'])
    
    # These should be the actual sparse grid points for training
    train_params = np.random.rand(n_train, dim) 
    test_params = np.random.rand(n_test, dim)
    
    # Add dummy columns for growth rate/frequency for compatibility
    train_params_full = np.hstack([train_params, np.random.rand(n_train, 2)])
    test_params_full = np.hstack([test_params, np.random.rand(n_test, 2)])

    np.savetxt(base_data_dir / config['training_params_file'], train_params_full)
    np.savetxt(base_data_dir / config['testing_params_file'], test_params_full)
    print(f"  - Created dummy parameter files.")
    
    print("\nDummy data generation complete.")
    print("You can now run the workflow.")

if __name__ == '__main__':
    main()