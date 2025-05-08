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
                           214661930, 214662807, 214662895, 214662905, 214664699, 214664842, 270185125, 270185254,
                           270187139, 270187208, 270187283]


def select_best_tic_ids(phot_table, args):
    """
    Select the best TIC_IDs based on the best linear model fit (lowest residual sum of squares),
    and reject stars with R-values much larger than the smallest R-value.
    """
    # Filter stars based on brightness range
    phot_table = phot_table[(phot_table['Tmag'] >= args.bl) & (phot_table['Tmag'] <= args.fl)]

    unique_tic_ids = np.unique(phot_table['TIC_ID'])
    print(f"Total stars in brightness range: {len(unique_tic_ids)}")

    stars_fit_list = []
    for tic_id in unique_tic_ids:
        star_data = phot_table[phot_table['TIC_ID'] == tic_id]
        rel_flux = star_data['Relative_Flux']
        time = star_data['Time_BJD']

        # Fit a linear model: y = mx + b
        slope, intercept, r_value, p_value, std_err = linregress(time, rel_flux)

        # Calculate residuals and residual sum of squares (RSS)
        fitted_line = slope * time + intercept
        rss = np.sum((rel_flux - fitted_line) ** 2)

        # Store TIC_ID, RSS, and R-value (absolute value to evaluate fit quality)
        stars_fit_list.append((tic_id, rss, abs(r_value)))

    # Find the smallest R-value (worst fit)
    smallest_r_value = min(stars_fit_list, key=lambda x: x[2])[2]
    print(f"Smallest R-value: {smallest_r_value:.6f}")

    # Set a rejection threshold: reject stars with R-values > 2 * smallest_r_value
    rejection_threshold = args.r * smallest_r_value
    print(f"Rejection threshold for R-values: {rejection_threshold:.6f}")

    # Filter stars by RSS and R-value
    filtered_stars = [
        (tic_id, rss, r_value) for tic_id, rss, r_value in stars_fit_list if r_value <= rejection_threshold
    ]

    # Sort by RSS (ascending order) and select the top `num_stars`
    sorted_stars = sorted(filtered_stars, key=lambda x: x[1])[:args.num_stars]
    best_tic_ids = [star[0] for star in sorted_stars]

    # Print each selected star along with its RSS and R-value
    print("\nSelected Stars with RSS and R-values:")
    for star_id, rss_value, r_value in sorted_stars:
        print(f"TIC_ID: {star_id}, RSS: {rss_value:.6f}, R-value: {r_value:.6f}")
    print(f'The best tic_ids are: {best_tic_ids}')
    return best_tic_ids


def filter_to_tic_ids(phot_table,  tic_ids):
    """Filter the photometry table to include only the specified TIC_IDs."""
    phot_table = phot_table[np.isin(phot_table['TIC_ID'], tic_ids)]
    return phot_table


def compute_rms_values(phot_table, exp, args):
    """Compute RMS values for the provided photometry table."""
    phot_table = phot_table[(phot_table['Tmag'] >= args.bl) & (phot_table['Tmag'] <= args.fl)]

    unique_tmags = np.unique(phot_table['Tmag'])
    print(f"Total stars in brightness range: {len(unique_tmags)}")

    average_rms_values = []
    times_binned = []
    max_binning = int(args.bin)

    for Tmag in unique_tmags:
        Tmag_data = phot_table[phot_table['Tmag'] == Tmag]
        tic_id = Tmag_data['TIC_ID']
        jd_mid = Tmag_data['Time_BJD']
        rel_flux = Tmag_data['Relative_Flux']
        rel_fluxerr = Tmag_data['Relative_Flux_err']
        RMS_data = np.array(np.std(rel_flux))
        color = Tmag_data['COLOR']
        print(f'Star {tic_id[0]}, color {color[0]}, and Tmag {Tmag}, and RMS: {RMS_data}')
        RMS_values = []
        time_seconds = []
        print(f'Using exposure time: {exp}')

        for i in range(1, max_binning):
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
            exposure_time_seconds = i * exp
            RMS = np.std(dt_flux_binned)
            RMS_values.append(RMS)
            time_seconds.append(exposure_time_seconds)

        average_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

    average_rms_values = np.median(average_rms_values, axis=0) * 1e6  # Convert to ppm
    times_binned = times_binned[0]  # Use the first time bin set

    binning_times = np.array([i for i in range(1, max_binning)])

    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    return times_binned, average_rms_values, RMS_model


def plot_two_rms(times1, avg_rms1, RMS_model1, times2, avg_rms2, RMS_model2, label1, label2):
    """
    Generate a single RMS plot for two datasets in one figure.

    Parameters:
        times1, times2 (array-like): Exposure times for the two datasets.
        avg_rms1, avg_rms2 (array-like): Average RMS values for the two datasets.
        RMS_model1, RMS_model2 (array-like): RMS models for the two datasets.
        label1, label2 (str): Labels for the two datasets.
    """
    # Create a single plot
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot for dataset 1
    ax.plot(times1, avg_rms1, 'o', label=f"{label1} Data", color='blue')
    ax.plot(times1, RMS_model1, '--', label=f"{label1} Model", color='blue')

    # Plot for dataset 2
    ax.plot(times2, avg_rms2, 'o', label=f"{label2} Data", color='red')
    ax.plot(times2, RMS_model2, '--', label=f"{label2} Model", color='red')

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
    plt.show()

    # Save results to a JSON file
    results = {
        "file1": {
            "label": 'CMOS',
            "times": list(times1),
            "avg_rms": list(avg_rms1),
            "rms_model": list(RMS_model1)
        },
        "file2": {
            "label": 'CCD',
            "times": list(times2),
            "avg_rms": list(avg_rms2),
            "rms_model": list(RMS_model2)
        }
    }

    with open("rms_vs_timescale.json", "w") as outfile:
        json.dump(results, outfile, indent=4)
    print("RMS vs Timescale results saved to rms_vs_timescale.json")


def process_file(phot_file, args):
    """Process a single photometry file."""
    print(f"Processing {phot_file}...")
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    print(f"Completed processing {phot_file}.")
    return phot_table


def downsample_phot_table(phot_table, step):
    """
    Downsample the photometry table by removing one data point for every `step` images.
    """
    indices_to_keep = [i for i in range(len(phot_table)) if (i + 1) % step != 0]
    return phot_table[indices_to_keep]


def plot_flux_histogram(phot_table1, phot_table2, label1, label2):
    """
    Plot overlaid histograms of relative flux distributions for the two photometry files.
    """
    # Extract relative flux values
    rel_flux1 = phot_table1['Relative_Flux']
    rms1 = phot_table1['RMS'][0]
    rel_flux2 = phot_table2['Relative_Flux']
    rms2 = phot_table2['RMS'][0]

    print(f'The size of rel flux1 and rel_flux2: {len(rel_flux1)}, {len(rel_flux2)}')

    # Create the histogram plot
    plt.figure()
    plt.hist(rel_flux1, bins=50, alpha=0.5, label=f'{label1}, RMS={rms1:.4f}', color='blue', density=True)
    plt.hist(rel_flux2, bins=50, alpha=0.5, label=f'{label2}, RMS={rms2:.4f} ', color='red', density=True)

    plt.xlabel('Relative Flux')
    plt.ylabel('Frequency')
    plt.title('Relative Flux Histogram Distribution')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def filter_by_color(phot_table, cl, ch):
    """
    Filter the photometry table by the color index range.
    """
    return phot_table[(phot_table['COLOR'] >= cl) & (phot_table['COLOR'] <= ch)]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and plot RMS for two files.')
    parser.add_argument('file1', type=str, help='Path to the first photometry file')
    parser.add_argument('file2', type=str, help='Path to the second photometry file')
    parser.add_argument('--num_stars', type=int, default=10, help='Number of stars to select')
    parser.add_argument('--bl', type=float, default=9.5, help='Lower limit for Tmag')
    parser.add_argument('--fl', type=float, default=10.5, help='Upper limit for Tmag')
    parser.add_argument('--cl', type=float, default=None, help='Lower limit for color index')
    parser.add_argument('--ch', type=float, default=None, help='Upper limit for color index')
    parser.add_argument('--bin', type=float, default=600, help='Maximum binning time in seconds')
    parser.add_argument('--r', type=float, default=2, help='Rejection multiplication')
    parser.add_argument('--best', action='store_true', help='Use predefined best TIC_IDs')
    parser.add_argument('--tic', type=int, default=None, help='plot individual TIC_ID')
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

    # Apply color filtering if limits are provided
    if args.cl is not None and args.ch is not None:
        phot_table1 = filter_by_color(phot_table1, args.cl, args.ch)
        phot_table2 = filter_by_color(phot_table2, args.cl, args.ch)

    # Select best TIC_IDs from the first file
    if args.best:
        print("Using predefined best TIC_IDs...")
        best_tic_ids = PREDEFINED_BEST_TIC_IDS
    else:
        print("Selecting best TIC_IDs based on criteria...")
        best_tic_ids = select_best_tic_ids(phot_table1, args)

    if args.tic is not None:
        best_tic_ids = [args.tic]

    # Filter both files to include only the best TIC_IDs
    phot_table1 = filter_to_tic_ids(phot_table1, best_tic_ids)
    phot_table2 = filter_to_tic_ids(phot_table2, best_tic_ids)

    times1, avg_rms1, RMS_model1 = compute_rms_values(phot_table1, 10, args)
    times2, avg_rms2, RMS_model2 = compute_rms_values(phot_table2, 13, args)

    if args.best:
        for tic in best_tic_ids:
            plot_two_rms(times1, avg_rms1, RMS_model1, times2, avg_rms2, RMS_model2, label1=args.file1, label2=args.file2)
    # plot_flux_histogram(phot_table1, phot_table2, label1='CMOS', label2='CCD')
    else:
        plot_two_rms(times1, avg_rms1, RMS_model1, times2, avg_rms2, RMS_model2, label1=args.file1, label2=args.file2)

