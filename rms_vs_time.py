#! /usr/bin/env python
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import (plot_images, find_current_night_directory, get_phot_files,
                   read_phot_file, bin_time_flux_error, remove_outliers, calculate_trend_and_flux)

# use dark background for plots
# plt.style.use('dark_background')


def plot_rms_time(table, num_stars, tic_id=None):
    filtered_table = table[(table['Tmag'] >= 9.2) & (table['Tmag'] <= 9.6)]
    unique_tmags = np.unique(filtered_table['Tmag'])
    print('The bright stars are: ', len(unique_tmags))

    average_rms_values = []
    times_binned = []
    max_binning = 151

    num_stars_used = 0
    num_stars_excluded = 0

    for Tmag in unique_tmags:
        # Get data for the current Tmag
        Tmag_data = table[table['Tmag'] == Tmag]
        # Extract relevant data
        jd_mid = Tmag_data['jd_mid']
        flux_5 = Tmag_data['flux_6']
        fluxerr_5 = Tmag_data['fluxerr_6']
        current_tic_id = Tmag_data['tic_id'][0]  # Assuming Tmag is the same for all jd_mid values of a star

        # Check if tic_id is specified and matches current_tic_id
        if tic_id is not None and current_tic_id != tic_id:
            continue

        time_clipped, flux_5_clipped, fluxerr_5_clipped, _, _ = remove_outliers(jd_mid, flux_5, fluxerr_5)

        trend, dt_flux, dt_fluxerr = calculate_trend_and_flux(time_clipped, flux_5_clipped, fluxerr_5_clipped)

        RMS_values = []
        time_seconds = []
        for i in range(1, max_binning):
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux, dt_fluxerr, i)
            exposure_time_seconds = i * 10  # 10 seconds per binning
            RMS = np.std(dt_flux_binned)
            RMS_values.append(RMS)
            time_seconds.append(exposure_time_seconds)

        # Check if the first RMS value is greater than 0.0065
        # if RMS_values[0] > 0.006:
        #     print('Excluding tic_id = {} and Tmag = {:.2f} due to RMS > 6000 ppm'.format(current_tic_id, Tmag))
        #     num_stars_excluded += 1
        #     continue
        else:
            print('Using star with tic_id = {} and Tmag = {:.2f} and RMS = {:.4f}'.
                  format(current_tic_id, Tmag, RMS_values[0]))

        num_stars_used += 1
        average_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

        # Stop if the number of stars used reaches the specified number
        if num_stars_used >= num_stars:
            break

    if not average_rms_values:
        print("No stars found. Skipping this photometry file.")
        return

    print('The bright stars are: {}, Stars used: {}, Stars excluded: {}'.format(
        len(unique_tmags), num_stars_used, num_stars_excluded))

    # Calculate the average RMS across all stars for each bin
    average_rms_values = np.mean(average_rms_values, axis=0) * 1000000  # Convert to ppm

    # Generate binning times
    binning_times = [i for i in range(1, max_binning)]

    # Calculate the expected decrease in RMS
    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    # Plot RMS as a function of exposure time along with the expected decrease in RMS
    plt.figure(figsize=(6, 10))
    plt.plot(times_binned[0], average_rms_values, 'o', color='blue', label='Actual RMS')
    plt.plot(times_binned[0], RMS_model, '--', color='red', label='Model RMS')
    plt.title('Time binned for file: ' + phot_file[-16:-4])
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


def run_for_more(phot_file, num_stars):
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    plot_rms_time(phot_table, num_stars)


if __name__ == "__main__":
    # Get the current night directory
    current_night_directory = '.'

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('--num_stars', type=int, default=0, help='Number of stars to plot')
    parser.add_argument('--tic_id', type=int, help='plot the time vs. binned RMS for a particular star')
    args = parser.parse_args()

    # Run the main function for each photometry file
    if args.tic_id is not None:
        for phot_file in phot_files:
            # main(phot_file, args.tic_id)
            run_for_one(phot_file, args.tic_id)
    else:
        for phot_file in phot_files:
            # main(phot_file, args.num_stars)
            run_for_more(phot_file, args.num_stars)
