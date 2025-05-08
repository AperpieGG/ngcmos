#! /usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, read_phot_file, bin_time_flux_error
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


def find_comp_star_rms(comp_tic_ids, phot_table):
    """
    Compute the RMS of relative flux for each comparison star TIC ID from the photometry table.

    Parameters:
        comp_tic_ids (list): List of comparison star TIC IDs.
        phot_table (astropy.table.Table or similar): Photometry table containing 'TIC_ID' and relative flux column.

    Returns:
        np.ndarray: Array of RMS values corresponding to the TIC IDs.
    """
    comp_star_rms = []
    print(f'The comp tic ids are: {comp_tic_ids}')

    for tic in comp_tic_ids:
        # Filter the table for this TIC ID
        mask = phot_table['TIC_ID'] == tic
        if np.sum(mask) == 0:
            raise ValueError(f"No data found for TIC ID {tic}")

        rel_flux = phot_table['Relative_Flux'][mask]
        if len(rel_flux) > 2:
            print(f'The length of the relative flux is {len(rel_flux)}.')
            comp_star_rms.append(np.array(np.std(rel_flux)[0]))
        else:
            raise ValueError(f"Multiple or no unique RMS values found for TIC ID {tic}: {np.std(rel_flux)}")

    return np.array(comp_star_rms)


def find_bad_comp_stars(comp_tic_ids, phot_table, comp_mags0, sig_level=4., dmag=0.25):
    # Calculate initial RMS of comparison stars
    comp_star_rms = find_comp_star_rms(comp_tic_ids, phot_table)
    print(f"Initial number of comparison stars: {len(comp_star_rms)}")

    comp_star_mask = np.ones(len(comp_star_rms), dtype=bool)
    i = 0

    while True:
        i += 1
        comp_mags = comp_mags0[comp_star_mask]
        comp_rms = comp_star_rms[comp_star_mask]
        N1 = len(comp_mags)

        if N1 == 0:
            print("No valid comparison stars left. Exiting.")
            break

        edges = np.arange(comp_mags.min(), comp_mags.max() + dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1] + edges[1:]) / 2.

        # Calculate median RMS per bin
        std_medians = np.array([np.median(comp_rms[dig == j]) if len(comp_rms[dig == j]) > 0 else np.nan
                                for j in range(1, len(edges))])

        # Remove NaNs from std_medians and mag_nodes
        valid_mask = ~np.isnan(std_medians)
        mag_nodes = mag_nodes[valid_mask]
        std_medians = std_medians[valid_mask]

        # Handle too few points for fitting
        if len(mag_nodes) < 4:
            if len(mag_nodes) > 1:
                mod = np.interp(comp_mags, mag_nodes, std_medians)
                mod0 = np.interp(comp_mags0, mag_nodes, std_medians)
            else:
                print("Not enough points for fitting. Exiting.")
                break
        else:
            spl = Spline(mag_nodes, std_medians, k=3)
            mod = spl(comp_mags)
            mod0 = spl(comp_mags0)

        std = np.std(comp_rms - mod)
        comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        N2 = np.sum(comp_star_mask)

        print(f"Iteration {i}: Stars included: {N2}, Stars excluded: {N1 - N2}")

        # Exit if the number of stars doesn't change or too many iterations
        if N1 == N2 or i > 11:
            break

    # Prepare data for visualization
    final_good_mask = comp_star_mask
    final_good_rms = comp_star_rms[final_good_mask]
    final_good_mags = comp_mags0[final_good_mask]

    final_bad_mask = ~comp_star_mask
    final_bad_rms = comp_star_rms[final_bad_mask]
    final_bad_mags = comp_mags0[final_bad_mask]

    # Determine plot limits based on the dimmest good star
    if len(final_good_rms) > 0:
        y_limit_high = 2 * max(final_good_rms)
        y_limit_low = min(final_good_rms) * 0.01
    else:
        y_limit_high, y_limit_low = 1, 0.01

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(final_good_mags, final_good_rms, color='black', label='Good Stars')
    plt.scatter(final_bad_mags, final_bad_rms, color='red', label='Bad Stars')
    plt.xlabel('Magnitude')
    plt.ylabel('RMS')
    plt.ylim(y_limit_low, y_limit_high)
    plt.legend()
    plt.title('RMS vs. Magnitude of Comparison Stars')
    plt.tight_layout()
    plt.show()

    print(f'RMS of comparison stars after filtering: {len(comp_star_rms[comp_star_mask])}')
    print(f'RMS values after filtering: {comp_star_rms[comp_star_mask]}')

    return comp_star_mask, comp_star_rms, i


def select_best_tic_ids(phot_table, args):
    # Filter by brightness
    phot_table = phot_table[(phot_table['Tmag'] >= args.bl) & (phot_table['Tmag'] <= args.fl)]

    # Identify unique comparison stars
    tic_ids = np.unique(phot_table['TIC_ID'])

    valid_tic_ids = []
    comp_mags = []

    for tic in tic_ids:
        star_data = phot_table[phot_table['TIC_ID'] == tic]
        if len(star_data) < 10:
            continue  # Skip stars with too little data
        valid_tic_ids.append(tic)
        comp_mags.append(star_data['Tmag'][0])

    valid_tic_ids = np.array(valid_tic_ids)
    comp_mags = np.array(comp_mags)

    comp_mask, comp_rms, iterations = find_bad_comp_stars(valid_tic_ids, phot_table, comp_mags)

    best_tic_ids = valid_tic_ids[comp_mask]
    print(f"Selected {len(best_tic_ids)} good comparison TIC IDs.")
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
        RMS_data = Tmag_data['RMS']
        color = Tmag_data['COLOR']
        print(f'Star {tic_id[0]}, color {color[0]}, and Tmag {Tmag}, and RMS: {RMS_data[0]}')
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
    parser.add_argument('--bl', type=float, default=9, help='Lower limit for Tmag')
    parser.add_argument('--fl', type=float, default=10, help='Upper limit for Tmag')
    parser.add_argument('--bin', type=float, default=180, help='Maximum binning time in seconds')
    args = parser.parse_args()

    # Process both files
    phot_table1 = process_file(args.file1, args)
    phot_table2 = process_file(args.file2, args)

    # the below statement is for testing!
    tic_id = 4611043
    mask = phot_table1['TIC_ID'] == tic_id
    if np.sum(mask) == 0:
        print(f"No data found for TIC ID {tic_id}")
    else:
        flux_length = len(phot_table1['Relative_Flux'][mask])
        print(f'The length of the relative flux for TIC ID {tic_id} in file1: {flux_length}')

    print("Trimming data in phot_table1")
    phot_table1 = trim_target_data(phot_table1)
    print("Trimming data in phot_table2")
    phot_table2 = trim_target_data(phot_table2)

    print("Trimming data in phot_table1 for time to exclude the twilight")
    phot_table1 = trim_target_data_by_time(phot_table1)
    print("Trimming data in phot_table2 for time to exclude the twilight")
    phot_table2 = trim_target_data_by_time(phot_table2)

    best_tic_ids = select_best_tic_ids(phot_table1, args)

    # Filter both files to include only the best TIC_IDs
    phot_table1 = filter_to_tic_ids(phot_table1, best_tic_ids)
    phot_table2 = filter_to_tic_ids(phot_table2, best_tic_ids)

    times1, avg_rms1, RMS_model1 = compute_rms_values(phot_table1, 10, args)
    times2, avg_rms2, RMS_model2 = compute_rms_values(phot_table2, 13, args)

    # plot_flux_histogram(phot_table1, phot_table2, label1='CMOS', label2='CCD')

    # Plot the results
    plot_two_rms(times1, avg_rms1, RMS_model1, times2, avg_rms2, RMS_model2, label1=args.file1, label2=args.file2)

