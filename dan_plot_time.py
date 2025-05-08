#! /usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, read_phot_file, bin_time_flux_error
from scipy.stats import linregress

PREDEFINED_BEST_TIC_IDS = [4611043, 5796255, 5796320, 5796376, 169746092, 169746369, 169746459, 169763609, 169763615,
                           169763631, 169763812, 169763929, 169763985, 169764011, 169764168, 169764174, 188619865,
                           188620052, 188620343, 188620450, 188620477, 188620644, 188622237, 188622268, 188622275,
                           188622523, 188627904, 188628115, 188628237, 188628252, 188628309, 188628413, 188628448,
                           188628555, 188628748, 188628755, 214657492, 214657985, 214658021, 214661588, 214661799,
                           214661930, 214662807, 214662895, 214662905, 214664699, 214664842,
                           270185125, 270185254, 270187139, 270187208, 270187283]


def filter_to_tic_ids(phot_table, tic_ids):
    """Filter the photometry table to include only the specified TIC_IDs."""
    phot_table = phot_table[np.isin(phot_table['TIC_ID'], tic_ids)]
    return phot_table


def process_file(phot_file, args):
    """Process a single photometry file."""
    print(f"Processing {phot_file}...")
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    print(f"Completed processing {phot_file}.")
    return phot_table


def trim_target_data_by_time(phot_table):
    """
    Trim the data points by removing the last 15 minutes of data based on `Time_BJD`.
    This is because we are including the astronomical twilight for both 20240705 and 20240622.
    This increases the red noise from morning light!

    :param phot_table: Input photometry table
    :return: Trimmed photometry table
    """
    unique_tmags = np.unique(phot_table['Tmag'])
    trimmed_table_list = []
    time_threshold = 15 / (24 * 60)  # 15 minutes in days

    for Tmag in unique_tmags:
        # Select data for the current target
        Tmag_data = phot_table[phot_table['Tmag'] == Tmag]

        # Find the last time in the data
        last_time = Tmag_data['Time_BJD'][-1]

        # Trim the data to exclude the last 15 minutes
        trimmed_data = Tmag_data[Tmag_data['Time_BJD'] <= (last_time - time_threshold)]

        # Add trimmed data to the list
        trimmed_table_list.append(trimmed_data)

    # Combine all trimmed targets back into a single table
    if len(trimmed_table_list) > 0:
        trimmed_table = np.hstack(trimmed_table_list)
    else:
        raise ValueError("No valid data points after trimming. Check your trimming criteria.")

    return trimmed_table


def trim_target_data(phot_table):
    """
    Trim the data points based on airmass criteria:
    - Trim from the beginning until the starting airmass is ≤ 1.75.
    - Trim from the end if the ending airmass exceeds 1.75.
    Print the total number of points trimmed from both the beginning and the end.

    :param phot_table: Input photometry table
    :return: Trimmed photometry table
    """
    unique_tmags = np.unique(phot_table['Tmag'])
    trimmed_table_list = []

    total_start_trim_count = 0
    total_end_trim_count = 0

    for Tmag in unique_tmags:
        # Select data for the current target
        Tmag_data = phot_table[phot_table['Tmag'] == Tmag]

        # Trim from the beginning until airmass ≤ 1.7
        start_trim_count = 0
        while start_trim_count < len(Tmag_data) and Tmag_data['Airmass'][start_trim_count] > 1.75:
            start_trim_count += 1
        trimmed_data = Tmag_data[start_trim_count:]

        # Trim from the end if airmass > 1.7
        end_trim_count = 0
        while len(trimmed_data) > 0 and trimmed_data['Airmass'][-1] > 1.75:
            end_trim_count += 1
            trimmed_data = trimmed_data[:-1]

        # Update total counts
        total_start_trim_count += start_trim_count
        total_end_trim_count += end_trim_count

        if len(trimmed_data) == 0:
            continue

        # Add trimmed data to the list
        trimmed_table_list.append(trimmed_data)

    # Print total trimming details
    print(f"Total points trimmed from the beginning: {total_start_trim_count}")
    print(f"Total points trimmed from the end: {total_end_trim_count}")

    # Combine all trimmed targets back into a single table
    if len(trimmed_table_list) > 0:
        trimmed_table = np.hstack(trimmed_table_list)
    else:
        raise ValueError("No valid data points after trimming. Check your trimming criteria.")

    return trimmed_table


def compute_rms_per_tic(tic_id, phot_table, exposure_time, args):
    """Compute RMS vs binning time for a single TIC ID."""
    star_data = phot_table[phot_table['TIC_ID'] == tic_id]

    if len(star_data) == 0:
        return None, None, None

    jd_mid = star_data['Time_BJD']
    rel_flux = star_data['Relative_Flux']
    rel_fluxerr = star_data['Relative_Flux_err']

    max_binning = int(args.bin)
    time_seconds = []
    rms_values = []

    for i in range(1, max_binning):
        time_binned, flux_binned, fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
        time_seconds.append(i * exposure_time)
        rms_values.append(np.std(flux_binned))

    rms_values = np.array(rms_values) * 1e6  # Convert to ppm
    rms_model = rms_values[0] / np.sqrt(np.arange(1, max_binning))

    return time_seconds, rms_values, rms_model


def plot_and_save_rms(t1, t2, rms1, rms2, model1, model2, tic_id):
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot for dataset 1
    ax.plot(t1, rms1, 'o', color='blue')
    ax.plot(t1, model1, '--', color='blue')

    # Plot for dataset 2
    ax.plot(t2, rms2, 'o', color='red')
    ax.plot(t2, model2, '--', color='red')

    # Add vertical line for reference
    ax.axvline(x=900, color='black', linestyle='-', label='Reference Line (x=900)')

    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add labels and legend
    ax.set_xlabel('Exposure Time (s)')
    ax.set_ylabel('RMS (ppm)')
    ax.set_title('RMS vs Exposure Time')
    # Format the y-axis tick labels
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.tick_params(axis='y', which='minor', length=4)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(f"time_{tic_id}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and plot RMS for two files.')
    parser.add_argument('file1', type=str, help='Path to the first photometry file')
    parser.add_argument('file2', type=str, help='Path to the second photometry file')
    parser.add_argument('--bl', type=float, default=9.5, help='Lower limit for Tmag')
    parser.add_argument('--fl', type=float, default=10.5, help='Upper limit for Tmag')
    parser.add_argument('--bin', type=float, default=180, help='Maximum binning time in seconds')
    args = parser.parse_args()

    # Process both files
    phot_table1 = process_file(args.file1, args)
    # phot_table1 = downsample_phot_table(phot_table1, step=3)
    phot_table2 = process_file(args.file2, args)

    print("Trimming data in phot_table1")
    phot_table1 = trim_target_data(phot_table1)
    print("Trimming data in phot_table2")
    phot_table2 = trim_target_data(phot_table2)

    print("Trimming data in phot_table1 for time")
    phot_table1 = trim_target_data_by_time(phot_table1)
    print("Trimming data in phot_table2 for time")
    phot_table2 = trim_target_data_by_time(phot_table2)

    best_tic_ids = PREDEFINED_BEST_TIC_IDS

    # Filter both files to include only the best TIC_IDs
    phot_table1 = filter_to_tic_ids(phot_table1, best_tic_ids)
    phot_table2 = filter_to_tic_ids(phot_table2, best_tic_ids)

    for tic_id in PREDEFINED_BEST_TIC_IDS:
        t1, rms1, model1 = compute_rms_per_tic(tic_id, phot_table1, 10, args)
        t2, rms2, model2 = compute_rms_per_tic(tic_id, phot_table2, 13, args)

        plot_and_save_rms(t1, t2, rms1, rms2, model1, model2, tic_id)


