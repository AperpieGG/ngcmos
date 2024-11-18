#! /usr/bin/env python
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, read_phot_file, bin_time_flux_error


def filter_to_common_targets(phot_table1, phot_table2):
    """Filter both photometry tables to include only common targets and return their TIC_IDs."""
    # Find the common targets based on TIC_ID
    common_targets = np.intersect1d(phot_table1['TIC_ID'], phot_table2['TIC_ID'])
    print(f"Number of common targets: {len(common_targets)}")

    # Filter each table to include only common targets
    phot_table1 = phot_table1[np.isin(phot_table1['TIC_ID'], common_targets)]
    phot_table2 = phot_table2[np.isin(phot_table2['TIC_ID'], common_targets)]

    return phot_table1, phot_table2, common_targets


def plot_two_rms(phot_table1, phot_table2, label1, label2, args):
    """Generate two RMS plots in a single figure with one row and two columns."""

    def compute_rms_values(phot_table, common_targets, args):
        """Compute RMS values for the provided photometry table using common targets."""
        # Filter to include only common targets
        phot_table = phot_table[np.isin(phot_table['TIC_ID'], common_targets)]

        # Apply magnitude filter
        phot_table = phot_table[(phot_table['Tmag'] >= args.bl) & (phot_table['Tmag'] <= args.fl)]

        unique_tmags = np.unique(phot_table['Tmag'])
        print(f"Total stars in brightness range: {len(unique_tmags)}")

        stars_rms_list = []
        for Tmag in unique_tmags:
            Tmag_data = phot_table[phot_table['Tmag'] == Tmag]
            rel_flux = Tmag_data['Relative_Flux']
            initial_rms = np.std(rel_flux)
            stars_rms_list.append((Tmag_data, initial_rms))

        sorted_stars = sorted(stars_rms_list, key=lambda x: x[1])[:args.num_stars]
        print(f"Selected {len(sorted_stars)} stars with lowest RMS values.")

        average_rms_values = []
        times_binned = []
        max_binning = int(args.bin)

        for Tmag_data, initial_rms in sorted_stars:
            jd_mid = Tmag_data['Time_BJD']
            rel_flux = Tmag_data['Relative_Flux']
            rel_fluxerr = Tmag_data['Relative_Flux_err']
            RMS_values = []
            time_seconds = []
            for i in range(1, max_binning):
                time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
                exposure_time_seconds = i * args.exp
                RMS = np.std(dt_flux_binned)
                RMS_values.append(RMS)
                time_seconds.append(exposure_time_seconds)

            average_rms_values.append(RMS_values)
            times_binned.append(time_seconds)

        average_rms_values = np.median(average_rms_values, axis=0) * 1e6  # Convert to ppm
        times_binned = times_binned[0]  # Use the first time bin set

        # Generate binning times
        binning_times = [i for i in range(1, max_binning)]

        # Expected RMS decrease model
        RMS_model = average_rms_values[0] / np.sqrt(binning_times)

        # Print selected TIC_IDs for verification
        print("TIC_IDs used for RMS calculation:")
        print(phot_table['TIC_ID'])

        return times_binned, average_rms_values, RMS_model

    # Compute RMS values for both files
    times1, avg_rms1, RMS_model1 = compute_rms_values(phot_table1)
    times2, avg_rms2, RMS_model2 = compute_rms_values(phot_table2)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
    axs[0].plot(times1, avg_rms1, 'o', label=label1, color='black')
    axs[0].plot(times1, RMS_model1, '--', color='black')
    axs[0].axvline(x=900, color='red', linestyle='-')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Exposure time (s)')
    axs[0].set_ylabel('RMS (ppm)')

    axs[1].plot(times2, avg_rms2, 'o', label=label2, color='black')
    axs[1].plot(times2, RMS_model2, '--', color='black')
    axs[1].axvline(x=900, color='red', linestyle='-')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Exposure time (s)')

    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().tick_params(axis='y', which='minor', length=4)

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()


def process_file(phot_file, args):
    """Process a single photometry file."""
    print(f"Processing {phot_file}...")
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    print(f"Completed processing {phot_file}.")
    return phot_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and plot RMS for two files.')
    parser.add_argument('file1', type=str, help='Path to the first photometry file')
    parser.add_argument('file2', type=str, help='Path to the second photometry file')
    parser.add_argument('--num_stars', type=int, default=5, help='Number of stars to plot (only if tic_id is not provided)')
    parser.add_argument('--bl', type=float, default=9.5, help='Lower limit for Tmag')
    parser.add_argument('--fl', type=float, default=10.5, help='Upper limit for Tmag')
    parser.add_argument('--exp', type=float, default=10.0, help='Exposure time in seconds')
    parser.add_argument('--bin', type=float, default=600, help='Maximum binning time in seconds')
    args = parser.parse_args()

    # Process files sequentially
    phot_table1 = process_file(args.file1, args)
    phot_table2 = process_file(args.file2, args)

    # Filter to common targets and get their TIC_IDs
    phot_table1, phot_table2, common_targets = filter_to_common_targets(phot_table1, phot_table2)

    # Compute RMS values for both files using the same targets
    times1, avg_rms1 = compute_rms_values(phot_table1, common_targets, args)
    times2, avg_rms2 = compute_rms_values(phot_table2, common_targets, args)

    # Plot results
    plot_two_rms(phot_table1, phot_table2, label1=args.file1, label2=args.file2, args=args)