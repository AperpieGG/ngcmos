#! /usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, read_phot_file, bin_time_flux_error
from scipy.stats import linregress

PREDEFINED_BEST_TIC_IDS = [214661857, 214661766, 214657879, 214664791, 214658017, 214664709, 188620487, 214657603,
                           214658150, 169764095, 214661742, 214664501, 188620442, 214661619, 169745976, 188628644,
                           214658115, 214661908, 188626100, 214664659, 214664822, 169764075, 188628293, 214658096,
                           214657567, 188620328, 188635755, 188625980, 214662758, 169746346, 169745834, 188620407,
                           214664671, 188620276, 188620486, 270185187, 5795978, 169763883, 169745813, 169764230,
                           188627979, 169764104, 214662768, 169746270, 188626048, 188619909, 214657837, 214662861,
                           270185349, 169746214, 169764293, 188620411, 188630638, 169745852, 188620043, 188628551,
                           214658071, 270185348, 214658053, 5796173, 214662808, 169763782, 188620284, 188619746,
                           188628772, 214657454, 270187137, 188626209, 188626046, 188619835, 169746225, 270185483,
                           169763860, 169763772, 188635802, 188625869, 188628050, 214661732, 214657448, 270187078,
                           188626087, 169763671, 214658075, 188635801, 169746381, 214664802, 214657785, 169745774,
                           214657480, 214657943, 214664788, 214663089, 214664503, 214661960, 169746262, 188630713,
                           270185219, 214661958, 214657833, 188620324, 214662724, 270185283, 270185107, 188630574,
                           214658951, 214661582, 5796179, 169745855, 188630514, 214664607, 188620003, 188628174,
                           270185456]


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


