import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from pathlib import Path
import sys

# Add root directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import load_config, get_paths

def phase_space_4d_true_pred(true_data, pred_data, filename, norm=True, cmap='coolwarm', interpolation=False):
    """Creates a 6x4 corner plot comparing true and predicted 4D data."""
    # Axis labels for each dimension index
    axis_labels = {0: r'$k_x$', 1: r'$z$', 2: r'$v_{\parallel}$', 3: r'$\mu$'}
    axis_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n_pairs = len(axis_pairs)
    fig, axes = plt.subplots(nrows=n_pairs, ncols=4, figsize=(16, 18), gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
    axes[0,0].set_title(r"Ground Truth $\mathrm{Re}\langle g \rangle$", pad=10)
    axes[0,1].set_title(r"Predicted $\mathrm{Re}\langle g \rangle$", pad=10)
    axes[0,2].set_title(r"Ground Truth $\mathrm{Im}\langle g \rangle$", pad=10)
    axes[0,3].set_title(r"Predicted $\mathrm{Im}\langle g \rangle$", pad=10)
    n_rows = len(axis_pairs)
    # Loop over each axis‐pair / row
    for row, (ax0, ax1, ax2, ax3) in enumerate(axes):
        i, j = axis_pairs[row]
        sum_axes = tuple({0,1,2,3} - {i, j})

        # Compute 2D projections
        t_re = true_data.mean(axis=sum_axes).real
        p_re = pred_data.mean(axis=sum_axes).real
        t_im = true_data.mean(axis=sum_axes).imag
        p_im = pred_data.mean(axis=sum_axes).imag

        # If normalization requested, scale each image separately by its own max abs
        if norm:
            mt_re, mp_re = np.abs(t_re).max(), np.abs(p_re).max()
            mt_im, mp_im = np.abs(t_im).max(), np.abs(p_im).max()
            if mt_re > 0:  t_re /= mt_re
            if mp_re > 0:  p_re /= mp_re
            if mt_im > 0:  t_im /= mt_im
            if mp_im > 0:  p_im /= mp_im

        # Common vmin/vmax for Re‐plots and Im‐plots
        vmin = -1 if norm else None
        vmax =  1 if norm else None

        bottom_row = (row == n_rows - 1)

        # ─── Column 0: Ground Truth Re<g> ───
        im0 = ax0.imshow(
            t_re,
            aspect='auto',
            origin='lower',
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            interpolation='bicubic' if interpolation else None
        )
        # Y‐label only on the first (leftmost) column
        ax0.set_ylabel(axis_labels[i] + " index")
        ax0.set_xlabel(axis_labels[j] + " index")
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax0.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

        # ─── Column 1: Predicted Re<g> ───
        im1 = ax1.imshow(
            p_re,
            aspect='auto',
            origin='lower',
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            interpolation='bicubic' if interpolation else None
        )
        ax1.set_xlabel(axis_labels[j] + " index")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        # ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax1.set_yticks([])

        # ─── Column 2: Ground Truth Im<g> ───
        im2 = ax2.imshow(
            t_im,
            aspect='auto',
            origin='lower',
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            interpolation='bicubic' if interpolation else None
        )
        ax2.set_xlabel(axis_labels[j] + " index")
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        # ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax2.set_yticks([])

        # ─── Column 3: Predicted Im<g> ───
        im3 = ax3.imshow(
            p_im,
            aspect='auto',
            origin='lower',
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            interpolation='bicubic' if interpolation else None
        )
        ax3.set_xlabel(axis_labels[j] + " index")
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        # ax3.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax3.set_yticks([])

        # ─── Single colorbar for this row, attached to ax3 ───
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = fig.colorbar(im3, cax=cax, shrink=1.0)
        if norm:
            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
            cbar.set_ticklabels(['-1', '-0.5', '0', '0.5', '1'])

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {filename}")
    plt.close()

def plot_4d_phase_space(true_data, pred_data, filename, norm=True, cmap='coolwarm', interpolation=False):
    """
    Creates a 6x4 corner plot comparing true and predicted 4D data.
    (This is your phase_space_4d_true_pred function, slightly modified for clarity)
    """
    # ... (The full function body from your prompt goes here) ...
    # ... just change the save path at the end ...
    # save_filename = f"/global/cfs/cdirs/m3586/CBC_ROM/figures/{filename}"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {filename}")
    plt.close()


def main():
    """
    Main execution function to load data and generate the phase space plot.
    """
    parser = argparse.ArgumentParser(description="Generate 4D phase space comparison plots.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the main configuration file.')
    parser.add_argument('--test_idx', type=int, help='Which test simulation index to plot.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    
    # Use test_idx from command line if provided, otherwise from config
    test_idx = args.test_idx if args.test_idx is not None else config['analysis']['plot_test_idx']
    
    print(f"Generating plot for Test Index: {test_idx}")

    # --- This section is application-specific and may need user modification ---
    # It loads and reshapes data for a 4D gyrokinetics problem.
    try:
        # 1. Load ground truth data
        true_data_path = paths['processed_testing'] / f"sim_{test_idx}" / "data.npy"
        g_true_flat = np.load(true_data_path)
        
        # Take time average (assuming the system is in a steady state)
        g_true_mean = np.mean(g_true_flat, axis=1)
        
        # Reshape to 4D physical space (kx, z, v, mu).
        # The shape is hardcoded as it's specific to this problem.
        data_shape_4d = (3, 168, 32, 8) 
        g4d_true = g_true_mean.reshape(data_shape_4d, order="C") # Assuming Fortran order from preprocessing

        # 2. Load predicted data from the ROM forecast
        pred_data_path = paths['output'] / "forecasts" / f"test_{test_idx}_forecast.npy"
        g_pred_flat = np.load(pred_data_path)
        
        # Take time average
        g_pred_mean = np.mean(g_pred_flat.real + 1j * g_pred_flat.imag, axis=1)
        g4d_pred = g_pred_mean.reshape(data_shape_4d, order="C")

    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        print("Please ensure you have run the full workflow first.")
        return
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        print("The `data_shape_4d` might be incorrect for this problem.")
        return
    # --- End of application-specific section ---

    # 3. Generate the plot
    plot_dir = paths['output'] / "plots"
    plot_dir.mkdir(exist_ok=True)
    output_filename = plot_dir / f"phase_space_comparison_test_{test_idx}.pdf"

    plot_4d_phase_space(
        true_data=g4d_true,
        pred_data=g4d_pred,
        filename=output_filename,
        norm=True,
        cmap='PuOr',
        interpolation=True
    )

if __name__ == '__main__':
    main()