import numpy as np
import os
import argparse
from pydmd import BOPDMD, DMD
from utils import load_config, get_paths, load_snapshots

def main(config_path):
    """
    For each training simulation, projects data onto the global basis and
    fits a local DMD model to find its spectral representation (eigenvalues, etc.).
    """
    config = load_config(config_path)
    paths = get_paths(config)
    output_path = paths['output']
    os.makedirs(output_path, exist_ok=True)
    
    r = config['rom_rank']

    # 1. Load the global POD basis
    U = np.load(output_path / "pod_basis_U.npy")
    V_global = U[:, :r]
    print(f"Loaded global basis of rank {r}. Shape: {V_global.shape}")

    # 2. Load snapshot data for fitting
    X_data, t_data = load_snapshots(
        paths['processed_training'],
        config['data_params']['training_indices'],
        config['data_params']
    )

    # 3. Fit a BOPDMD model for each simulation
    n_sims = len(X_data)
    eigenvalues_list = []
    eigenvectors_list = []
    amplitudes_list = []

    print("Fitting local DMD models for each training parameter...")
    for i, (X_sim, t_sim) in enumerate(zip(X_data, t_data)):
        print(f"  - Fitting model for simulation {i+1}/{n_sims}...")
        
        # Use a standard DMD to get a good initial guess for frequencies
        dmd0 = DMD(svd_rank=r)
        dmd0.fit(X_sim)
        dt = t_sim[1] - t_sim[0] if len(t_sim) > 1 else 1.0
        init_alpha = np.log(dmd0.eigs) / dt

        # Fit the Bagging/Optimized DMD on the projected data
        dmd_model = BOPDMD(
            svd_rank=r,
            num_trials=0,
            proj_basis=V_global,
            use_proj=True,
            eig_sort='abs',
            init_alpha=init_alpha,
        )
        dmd_model.fit(X_sim, t=t_sim)
        
        eigvals_reduced = dmd_model.eigs
        eigvecs_reduced = V_global.conj().T @ dmd_model.modes # shape (r, r)
        amplitudes = dmd_model._b # initial conditions

        eigenvalues_list.append(eigvals_reduced)
        eigenvectors_list.append(eigvecs_reduced)
        amplitudes_list.append(amplitudes)

    # 4. Save the spectral data
    np.save(output_path / "global_basis.npy", V_global) # Final basis used
    np.save(output_path / "training_eigenvalues.npy", np.array(eigenvalues_list))
    np.save(output_path / "training_eigenvectors.npy", np.array(eigenvectors_list))
    np.save(output_path / "training_amplitudes.npy", np.array(amplitudes_list))
    
    print(f"\nSaved training spectral data to {output_path}")
    print(f"  - Eigenvalues shape: {np.array(eigenvalues_list).shape}")
    print(f"  - Eigenvectors shape: {np.array(eigenvectors_list).shape}")
    print(f"  - Amplitudes shape: {np.array(amplitudes_list).shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit local ROMs to training data.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)