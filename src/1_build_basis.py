import numpy as np
import os
import argparse
from utils import load_config, get_paths, load_snapshots
from sklearn.utils.extmath import randomized_svd

def main(config_path):
    """
    Builds a global POD basis from a collection of training snapshots.
    """
    config = load_config(config_path)
    paths = get_paths(config)
    os.makedirs(paths['output'], exist_ok=True)

    # 1. Load snapshot data using the utility function
    X_data, _ = load_snapshots(
        paths['processed_training'],
        config['data_params']['training_indices'],
        config['data_params']
    )

    # 2. Build the big snapshot matrix
    big_matrix = np.hstack(X_data)
    del X_data
    print(f"Constructed big matrix with shape: {big_matrix.shape}")

    # 3. Perform SVD to get the global basis
    print("Performing SVD on the big matrix...")
    # Using randomized_svd can be much faster for large matrices
    # U, S, Vh = randomized_svd(big_matrix, n_components=config['rom_rank'] * 5, random_state=42)
    U, S, Vh = np.linalg.svd(big_matrix, full_matrices=False)
    print(f"SVD complete. Shape of U: {U.shape}")

    # 4. Save the basis, singular values, and right singular vectors
    output_path = paths['output']
    np.save(output_path / "pod_basis_U.npy", U)
    np.save(output_path / "singular_values_S.npy", S)
    np.save(output_path / "right_singular_vectors_Vh.npy", Vh)
    print(f"Saved POD basis and singular values to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a global POD basis.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)