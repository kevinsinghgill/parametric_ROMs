import numpy as np
import argparse
import os
from utils import load_config, get_paths

def read_gyrokinetics_binary(input_filename, data_shape, n_time):
    """
    Reads a specific binary format from a gyrokinetics simulation.
    This function can be replaced by users with their own data readers.
    """
    N = np.prod(data_shape)
    times = np.empty(n_time, dtype=np.float64)
    data = np.empty(data_shape + (n_time,), dtype=np.complex128, order='F')

    with open(input_filename, 'rb') as fstream:
        for t in range(n_time):
            # Assuming double precision (8 bytes)
            offset = t * (2 * N + 1)
            
            # Read time
            fstream.seek(offset * 8)
            times[t] = np.fromfile(fstream, dtype=np.float64, count=1)[0]

            # Read data
            fstream.seek((offset + 1) * 8)
            data_block = np.fromfile(fstream, dtype=np.float64, count=2 * N)
            
            data_real = data_block[0::2].reshape(data_shape, order='F')
            data_imag = data_block[1::2].reshape(data_shape, order='F')
            data[..., t] = data_real + 1j * data_imag

    # Reshape data to a 2D matrix (space x time)
    data_matrix = data.reshape(N, n_time)
    return data_matrix, times

def main(config_path):
    """
    Main preprocessing script. Reads raw data, processes it, and saves it
    in a standardized format (.npy files).
    """
    config = load_config(config_path)
    paths = get_paths(config)

    print("Starting data preprocessing...")
    
    # Example for processing training data
    # NOTE: This part is highly specific to the original user's setup.
    # It assumes a mapping between `training_indices` and `n_times_list`.
    # A more robust solution would be a manifest file.
    data_shape = tuple(config['data_params']['data_shape'])
    n_times_list = config['data_params']['n_times_list']

    for i, idx in enumerate(config['data_params']['training_indices']):
        input_file = paths['raw_training'] / f"ETG_sim_{idx}/g1.dat"
        output_dir = paths['processed_training'] / f"sim_{idx}"
        os.makedirs(output_dir, exist_ok=True)
        
        n_time = n_times_list[i] # This mapping is brittle!
        print(f"Processing {input_file}...")

        data_matrix, times = read_gyrokinetics_binary(input_file, data_shape, n_time)

        np.save(output_dir / "data.npy", data_matrix)
        np.save(output_dir / "times.npy", times)

    print("Preprocessing complete.")
    # Add a similar loop for testing data if needed.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw simulation data.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)