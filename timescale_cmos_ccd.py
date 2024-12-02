#! /usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, read_phot_file, bin_time_flux_error


def select_best_tic_ids(phot_table, args):
    """
    Select the best TIC_IDs based on the flattest flux values (lowest residual sum of squares).
    """
    phot_table = phot_table[(phot_table['Tmag'] >= args.bl) & (phot_table['Tmag'] <= args.fl)]

    unique_tmags = np.unique(phot_table['Tmag'])
    print(f"Total stars in brightness range: {len(unique_tmags)}")

    stars_rss_list = []
    for Tmag in unique_tmags:
        Tmag_data = phot_table[phot_table['Tmag'] == Tmag]
        rel_flux = Tmag_data['Relative_Flux']
        mean_flux = np.mean(rel_flux)
        rss = np.sum((rel_flux - mean_flux) ** 2)  # Calculate Residual Sum of Squares (RSS)

        # Store TIC_ID and RSS if RSS < 0.095
        if rss < 0.095:
            stars_rss_list.append((Tmag_data['TIC_ID'][0], rss))

    # Print all selected stars and their RSS values
    print("\nSelected Stars with RSS < 0.095:")
    for star_id, rss_value in stars_rss_list:
        print(f"TIC_ID: {star_id}, RSS: {rss_value:.6f}")

    # Extract TIC_IDs
    best_tic_ids = [star[0] for star in stars_rss_list]
    print(f"\nNumber of selected stars: {len(best_tic_ids)}")

    return best_tic_ids


def filter_to_tic_ids(phot_table, tic_ids):
    """Filter the photometry table to include only the specified TIC_IDs."""
    phot_table = phot_table[np.isin(phot_table['TIC_ID'], tic_ids)]
    return phot_table


def compute_rms_values(phot_table, args):
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
    print(f'The times binned is: {times_binned[0]}')

    binning_times = np.array([i for i in range(1, max_binning)])

    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    return times_binned, average_rms_values, RMS_model


def plot_two_rms(times1, avg_rms1, RMS_model1, times2, avg_rms2, RMS_model2, label1, label2):
    """Generate two RMS plots in a single figure with one row and two columns."""
    fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharey=True)

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


def trim_target_data(phot_table, trim_count):
    """
    Trim the specified number of data points from the beginning and end for each target's data.
    :param phot_table: Input photometry table
    :param trim_count: Number of data points to remove from the beginning and the end
    :return: Trimmed photometry table
    """
    unique_tmags = np.unique(phot_table['Tmag'])
    trimmed_table_list = []

    for Tmag in unique_tmags:
        # Select data for the current target
        Tmag_data = phot_table[phot_table['Tmag'] == Tmag]
        if len(Tmag_data) <= 2 * trim_count:
            print(f"Skipping target with Tmag {Tmag} due to insufficient data points.")
            continue

        # Trim data points for the target
        trimmed_data = Tmag_data[trim_count:-trim_count]

        trimmed_table_list.append(trimmed_data)

    # Combine all trimmed targets back into a single table
    if len(trimmed_table_list) > 0:
        trimmed_table = np.hstack(trimmed_table_list)
    else:
        raise ValueError("No valid data points after trimming. Check your trim count.")

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
    parser.add_argument('--exp', type=float, default=10.0, help='Exposure time in seconds')
    parser.add_argument('--bin', type=float, default=600, help='Maximum binning time in seconds')
    parser.add_argument('--trim', type=int,
                        help='Number of data points to trim from the beginning and end')
    args = parser.parse_args()

    # Process both files
    phot_table1 = process_file(args.file1, args)
    # phot_table1 = downsample_phot_table(phot_table1, step=3)
    phot_table2 = process_file(args.file2, args)

    # Trim data for each target in both photometry tables
    if args.trim:
        print("Trimming data in phot_table1")
        phot_table1 = trim_target_data(phot_table1, args.trim)
        print("Trimming data in phot_table2")
        phot_table2 = trim_target_data(phot_table2, args.trim)

    # Apply color filtering if limits are provided
    if args.cl is not None and args.ch is not None:
        phot_table1 = filter_by_color(phot_table1, args.cl, args.ch)
        phot_table2 = filter_by_color(phot_table2, args.cl, args.ch)

    # Select best TIC_IDs from the first file
    best_tic_ids = select_best_tic_ids(phot_table1, args)

    # Filter both files to include only the best TIC_IDs
    phot_table1 = filter_to_tic_ids(phot_table1, best_tic_ids)
    phot_table2 = filter_to_tic_ids(phot_table2, best_tic_ids)

    # Compute RMS values for both files
    times1, avg_rms1, RMS_model1 = compute_rms_values(phot_table1, args)
    times2, avg_rms2, RMS_model2 = compute_rms_values(phot_table2, args)

    # plot_flux_histogram(phot_table1, phot_table2, label1='CMOS', label2='CCD')
    
    # Plot the results
    plot_two_rms(times1, avg_rms1, RMS_model1, times2, avg_rms2, RMS_model2, label1=args.file1, label2=args.file2)

