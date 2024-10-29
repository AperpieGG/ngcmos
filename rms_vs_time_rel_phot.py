#! /usr/bin/env python
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, get_rel_phot_files, read_phot_file, bin_time_flux_error


def plot_rms_time(table, num_stars, lower_limit, upper_limit, tic_id=None):
    # Filter by Tmag range
    filtered_table = table[(table['Tmag'] >= lower_limit) & (table['Tmag'] <= upper_limit)]
    unique_tmags = np.unique(filtered_table['Tmag'])
    print('Total stars in brightness range:', len(unique_tmags))

    # Initialize arrays to hold star data
    stars_rms_list = []

    # Iterate over each Tmag to calculate the initial RMS for each star
    for Tmag in unique_tmags:
        Tmag_data = filtered_table[filtered_table['Tmag'] == Tmag]
        rel_flux = Tmag_data['Relative_Flux']

        # Calculate initial RMS and append it to list
        initial_rms = np.std(rel_flux)
        stars_rms_list.append((Tmag_data, initial_rms))

    # Sort stars by initial RMS value and select top `num_stars`
    sorted_stars = sorted(stars_rms_list, key=lambda x: x[1])[:num_stars]
    print(f'Selected {len(sorted_stars)} stars with lowest RMS values.')

    # Prepare data for plotting
    average_rms_values = []
    times_binned = []
    max_binning = 151

    for Tmag_data, initial_rms in sorted_stars:
        jd_mid = Tmag_data['Time_BJD']
        rel_flux = Tmag_data['Relative_Flux']
        rel_fluxerr = Tmag_data['Relative_Flux_err']
        current_tic_id = Tmag_data['TIC_ID'][0]

        RMS_values = []
        time_seconds = []
        for i in range(1, max_binning):
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
            exposure_time_seconds = i * 10  # 10 seconds per bin
            RMS = np.std(dt_flux_binned)
            RMS_values.append(RMS)
            time_seconds.append(exposure_time_seconds)

        average_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

        # Print details for each selected star
        print(f'Star TIC_ID = {current_tic_id}, Tmag = {Tmag_data["Tmag"][0]}, '
              f'Initial RMS = {initial_rms:.4f}, Final RMS = {RMS_values[0]:.4f}')

    if not average_rms_values:
        print("No stars found. Skipping this photometry file.")
        return

    # Calculate average RMS across all selected stars
    average_rms_values = np.mean(average_rms_values, axis=0) * 1e6  # Convert to ppm

    # Generate binning times
    binning_times = [i for i in range(1, max_binning)]

    # Expected RMS decrease model
    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    # Plot RMS over time
    plt.figure(figsize=(6, 10))
    plt.plot(times_binned[0], average_rms_values, 'o', color='blue', label='Actual RMS')
    plt.plot(times_binned[0], RMS_model, '--', color='red', label='Model RMS')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposure time (s)')
    plt.ylabel('RMS (ppm)')

    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().tick_params(axis='y', which='minor', length=4)
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_for_one(phot_file, tic_id=None):
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    plot_rms_time(phot_table, 5, tic_id)


def run_for_more(phot_file, num_stars, lower_limit, upper_limit):
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    plot_rms_time(phot_table, num_stars, lower_limit, upper_limit)


if __name__ == "__main__":
    current_night_directory = '.'
    phot_files = get_rel_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('--num_stars', type=int, default=0, help='Number of stars to plot')
    parser.add_argument('--tic_id', type=int, help='plot the time vs. binned RMS for a particular star')
    parser.add_argument('--lower_limit', type=float, default=10.5, help='Lower limit for Tmag')
    parser.add_argument('--upper_limit', type=float, default=11.5, help='Upper limit for Tmag')
    args = parser.parse_args()

    if args.tic_id is not None:
        for phot_file in phot_files:
            run_for_one(phot_file, args.tic_id)
    else:
        for phot_file in phot_files:
            run_for_more(phot_file, args.num_stars, args.lower_limit, args.upper_limit)