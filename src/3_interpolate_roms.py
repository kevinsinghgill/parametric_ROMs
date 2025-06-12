import numpy as np
import argparse
import os
import contextlib
from sg_lib.grid.grid import Grid
from sg_lib.algebraic.multiindex import Multiindex
from sg_lib.operation.interpolation_to_spectral import InterpolationToSpectral
from utils import (load_config, get_paths, get_param_cols_and_bounds,
                   normalize_params, denormalize_params, load_params)

def interpolate_quantity(data, grid_obj, multiindex_set, test_points):
    """
    Generic function to perform sparse grid interpolation on a given quantity.
    
    Args:
        data (np.ndarray): Data to interpolate. Shape (n_training_points, ...).
        grid_obj (Grid): The sparse grid object.
        multiindex_set: The multi-index set for the grid.
        test_points (np.ndarray): Points to interpolate to. Shape (n_test_points, dim).

    Returns:
        Interpolated data at test_points.
    """
    dim = grid_obj.dim
    level = grid_obj.level
    
    # This assumes Clenshaw-Curtis, change if using other point types
    level_to_nodes = 1
    weights = [lambda x: 1.0 for _ in range(dim)]
    bounds = grid_obj.bounds
    
    interp_obj = InterpolationToSpectral(dim, level_to_nodes, bounds[0], bounds[1], weights, level, grid_obj)
    
    # This part of sg_lib seems to require re-populating for each item.
    # We map training points to their values.
    # Note: This assumes the training parameter points correspond to sg_lib's grid points.
    # A more robust implementation would map a parameter vector to a value.
    training_points = grid_obj.get_sg_points()

    for i, point in enumerate(training_points):
        sg_val = data[i]
        interp_obj.update_sg_evals_all_lut(point, sg_val)

    # This seems to be a necessary step in sg_lib
    for multiindex in multiindex_set:
         interp_obj.update_sg_evals_multiindex_lut(multiindex, grid_obj)

    interpolator = lambda x: interp_obj.eval_operation_sg(multiindex_set, x)
    
    interpolated_values = np.array([interpolator(p) for p in test_points])
    return interpolated_values

def main(config_path):
    config = load_config(config_path)
    paths = get_paths(config)
    output_path = paths['output']
    sg_params = config['sg_interpolation']
    param_cfg = config['param_space']

    # 0. Load data required for interpolation
    eigenvalues_train = np.load(output_path / "training_eigenvalues.npy")
    eigenvectors_train = np.load(output_path / "training_eigenvectors.npy")
    amplitudes_train = np.load(output_path / "training_amplitudes.npy")
    n_train, r = eigenvalues_train.shape

    # 1. Setup Sparse Grid
    dim = param_cfg['param_dim']
    level = sg_params['sg_level']
    grid_obj = Grid(dim, level, sg_params['level_to_nodes'],
                    left_bounds=np.zeros(dim), right_bounds=np.ones(dim))
    multiindex_set = Multiindex(dim).get_std_total_degree_mindex(level)

    # Sanity check: ensure training data matches SG point count
    if n_train != grid_obj.get_sg_set_size():
        raise ValueError(
            f"Number of training points ({n_train}) does not match the number of "
            f"sparse grid points ({grid_obj.get_sg_set_size()}) for level {level}."
        )

    # 2. Prepare test points and ground truth data based on config
    print("Preparing test points and ground truth data...")
    param_cols, param_bounds = get_param_cols_and_bounds(config)
    
    test_params_phys_from_file, true_vals = load_params(
        paths['testing_params'],
        param_cols,
        true_val_cols=[-2, -1] # Assumes last two columns are ground truth
    )
    n_test = len(config['data_params']['testing_indices'])

    test_mode = param_cfg['testing_setup']['mode']
    if test_mode == 'generate':
        print("Mode: 'generate'. Creating random test points with fixed seed.")
        seed = param_cfg['testing_setup']['random_seed']
        np.random.seed(seed)
        test_params_norm = np.random.uniform(0, 1, size=(n_test, dim))
        test_params_phys = denormalize_params(test_params_norm, param_bounds)

    elif test_mode == 'load':
        print("Mode: 'load'. Normalizing physical parameters from file.")
        test_params_phys = test_params_phys_from_file
        test_params_norm = normalize_params(test_params_phys, param_bounds)
        if np.any(test_params_norm < 0) or np.any(test_params_norm > 1):
            print("WARNING: Some test parameters fall outside the specified 'param_bounds'.")
            print("         Interpolation may be less accurate (extrapolation).")
    else:
        raise ValueError(f"Invalid testing_setup mode: '{test_mode}'. Choose 'generate' or 'load'.")
    
    print(f"Prepared {n_test} test parameter sets.")

    # 3. Perform interpolation for each quantity
    # --- Interpolate Eigenvalues ---
    print("Interpolating eigenvalues...")
    eigenvalues_test = np.zeros((n_test, r), dtype=complex)
    for i in range(r):
        eigenvalues_test[:, i] = interpolate_quantity(
            eigenvalues_train[:, i], grid_obj, multiindex_set, test_params_norm
        )

    # --- Interpolate Eigenvectors ---
    print("Interpolating eigenvectors...")
    eigenvectors_test = np.zeros((n_test, r, r), dtype=complex)
    for i in range(r):
        for j in range(r):
            eigenvectors_test[:, i, j] = interpolate_quantity(
                eigenvectors_train[:, i, j], grid_obj, multiindex_set, test_params_norm
            )
            
    # --- Interpolate Amplitudes ---
    print("Interpolating amplitudes...")
    amplitudes_test = np.zeros((n_test, r), dtype=complex)
    for i in range(r):
        amplitudes_test[:, i] = interpolate_quantity(
            amplitudes_train[:, i], grid_obj, multiindex_set, test_params_norm
        )

    # 4. Save interpolated results
    np.save(output_path / "testing_eigenvalues.npy", eigenvalues_test)
    np.save(output_path / "testing_eigenvectors.npy", eigenvectors_test)
    np.save(output_path / "testing_amplitudes.npy", amplitudes_test)
    np.save(output_path / "testing_parameters_physical.npy", test_params_phys)
    np.save(output_path / "testing_parameters_normalized.npy", test_params_norm)
    print(f"\nSaved interpolated ROM components to {output_path}")

    # 5. Validation against ground truth
    print("\n--- Validation Against Ground Truth ---")
    output_file_path = output_path / "interpolation_validation_report.txt"
    with open(output_file_path, "w") as out_file:
        with contextlib.redirect_stdout(out_file):
            print(f"Report for run: {output_path.name}")
            avg_rel_err_growth, avg_rel_err_freq = [], []

            for i in range(n_test):
                interp_eigs = eigenvalues_test[i, :]
                sorted_eigs = interp_eigs[np.argsort(-interp_eigs.real)]
                predicted_eig = sorted_eigs[0]

                pred_growth = predicted_eig.real
                pred_freq = predicted_eig.imag
                
                true_growth = true_vals[i, 0]
                true_freq = true_vals[i, 1]

                rel_err_growth = np.abs(pred_growth - true_growth) / np.abs(true_growth) if true_growth != 0 else np.nan
                rel_err_freq = np.abs(pred_freq - true_freq) / np.abs(true_freq) if true_freq != 0 else np.nan
                
                if not np.isnan(rel_err_growth):
                    avg_rel_err_growth.append(rel_err_growth)
                if not np.isnan(rel_err_freq):
                    avg_rel_err_freq.append(rel_err_freq)

                print(f"\n--- Test Case {i+1} ---")
                print(f"Physical Params: {np.round(test_params_phys[i], 3)}")
                print(f"Predicted -> Growth: {pred_growth:.4f}, Freq: {pred_freq:.4f}")
                print(f"True      -> Growth: {true_growth:.4f}, Freq: {true_freq:.4f}")
                print(f"Relative Error -> Growth: {rel_err_growth:.2%}, Freq: {rel_err_freq:.2%}")

            print("\n--- Summary ---")
            print(f"Average Relative Error (Growth Rate): {np.mean(avg_rel_err_growth):.2%}")
            print(f"Average Relative Error (Frequency):   {np.mean(avg_rel_err_freq):.2%}")

    print(f"Validation report saved to {output_file_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpolate ROM properties and validate.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)