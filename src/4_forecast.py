import numpy as np
import os
import argparse
from pydmd import BOPDMD
from utils import load_config, get_paths

def main(config_path):
    """
    Uses interpolated ROM components to build a predictive model for each
    test parameter and generates a forecast.
    """
    config = load_config(config_path)
    paths = get_paths(config)
    output_path = paths['output']
    r = config['rom_rank']

    # 1. Load all necessary pre-computed and interpolated data
    V_global = np.load(output_path / "global_basis.npy")
    eigenvalues_test = np.load(output_path / "testing_eigenvalues.npy")
    eigenvectors_test = np.load(output_path / "testing_eigenvectors.npy")
    amplitudes_test = np.load(output_path / "testing_amplitudes.npy")

    print("Loaded global basis and interpolated spectral data.")

    # 2. Loop through each test case to build a model and forecast
    for i, test_idx in enumerate(config['data_params']['testing_indices']):
        print(f"\n--- Forecasting for Test Case {test_idx} ---")
        
        # Get the interpolated components for this specific test case
        eigvals = eigenvalues_test[i, :]
        eigvecs = eigenvectors_test[i, :, :]
        amps = amplitudes_test[i, :]

        # 3. Reconstruct the reduced-order operator Atilde
        # Atilde = V * Lambda * V^-1
        Atilde = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)

        # 4. Reconstruct the full-space DMD modes
        # Note: Normalizing the reduced-order eigenvectors is important
        # for the stability of the full-space modes.
        eigvecs_norm = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)
        modes_full = V_global @ eigvecs_norm

        # 5. "Frankenstein" a pydmd object with the interpolated components
        # We need to initialize it with dummy data to create the object structure.
        dmd = BOPDMD(svd_rank=r)
        dmd.fit(np.random.rand(V_global.shape[0], r + 1)) # Dummy fit

        # Now, overwrite the internal state with our interpolated values.
        # This is the core of the parametric prediction.
        dmd.operator._Atilde = Atilde
        dmd.operator._eigenvalues = np.exp(eigvals) # pydmd expects discrete-time eigs
        dmd.operator._modes = modes_full
        dmd._amplitudes = amps
        
        # 6. Load the timestamps for the test case and generate the forecast
        sim_path = paths['processed_testing'] / f"sim_{test_idx}"
        t_test = np.load(sim_path / "times.npy")
        
        dmd.dmd_time.t0 = t_test[0]
        dmd.dmd_time.tend = t_test[-1]
        dmd.dmd_time.dt = t_test[1] - t_test[0]
        
        pred = dmd.reconstructed_data
        
        print(f"  - Forecast generated with shape: {pred.shape}")

        # 7. Save the results
        forecast_dir = output_path / "forecasts"
        os.makedirs(forecast_dir, exist_ok=True)
        np.save(forecast_dir / f"test_{test_idx}_forecast.npy", pred)
        np.save(forecast_dir / f"test_{test_idx}_modes.npy", modes_full)
        np.save(forecast_dir / f"test_{test_idx}_Atilde.npy", Atilde)

    print(f"\nAll forecasts saved to {forecast_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Forecast dynamics for new parameters.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)