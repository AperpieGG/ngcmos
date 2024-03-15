#!/usr/bin/env python
import argparse
import datetime
import json
import os
import fnmatch
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from utils import plot_images


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    If not found, use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str
        Path to the current night directory.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    return current_date_directory if os.path.isdir(current_date_directory) else os.getcwd()


def get_phot_files(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'phot_*.fits'):
            phot_files.append(filename)
    return phot_files


def read_phot_file(filename):
    """
    Read the photometry file.

    Parameters
    ----------
    filename : str
        Photometry file to read.

    Returns
    -------
    astropy.table.table.Table
        Table containing the photometry data.
    """
    # Read the photometry file here using fits or any other appropriate method
    try:
        with fits.open(filename) as ff:
            # Access the data in the photometry file as needed
            tab = ff[1].data
            return tab
    except Exception as e:
        print(f"Error reading photometry file {filename}: {e}")
        return None


def bin_time_flux_error(time, flux, error, bin_fact):
    """
    Use reshape to bin light curve data, clip under filled bins
    Works with 2D arrays of flux and errors

    Note: under filled bins are clipped off the end of the series

    Parameters
    ----------
    time : array         of times to bin
    flux : array         of flux values to bin
    error : array         of error values to bin
    bin_fact : int
        Number of measurements to combine

    Returns
    -------
    times_b : array
        Binned times
    flux_b : array
        Binned fluxes
    error_b : array
        Binned errors

    Raises
    ------
    None
    """
    n_binned = int(len(time) / bin_fact)
    clip = n_binned * bin_fact
    time_b = np.average(time[:clip].reshape(n_binned, bin_fact), axis=1)
    # determine if 1 or 2d flux/err inputs
    if len(flux.shape) == 1:
        flux_b = np.average(flux[:clip].reshape(n_binned, bin_fact), axis=1)
        error_b = np.sqrt(np.sum(error[:clip].reshape(n_binned, bin_fact) ** 2, axis=1)) / bin_fact
    else:
        # assumed 2d with 1 row per star
        n_stars = len(flux)
        flux_b = np.average(flux[:clip].reshape((n_stars, n_binned, bin_fact)), axis=2)
        error_b = np.sqrt(np.sum(error[:clip].reshape((n_stars, n_binned, bin_fact)) ** 2, axis=2)) / bin_fact
    return time_b, flux_b, error_b


def plot_rms_time(table, num_stars):
    # Filter table for stars within desired Tmag range
    filtered_table = table[(table['Tmag'] >= 8) & (table['Tmag'] <= 9.1)]
    # filtered_table = table[(table['Tmag'] >= 7.5) & (table['Tmag'] <= 9.5)]

    # Sort the table by Tmag (brightness)
    unique_tmags = np.unique(filtered_table['Tmag'])
    print('The bright stars are: ', len(unique_tmags))

    # Take the ones which are on the argument
    filtered_table = filtered_table[:num_stars]

    average_rms_values = []
    times_binned = []
    max_binning = 151

    num_stars_used = 0

    for gaia_id in filtered_table['gaia_id']:  # Loop over all stars in the filtered table
        gaia_id_data = table[table['gaia_id'] == gaia_id]
        jd_mid = gaia_id_data['jd_mid']
        flux_3 = gaia_id_data['flux_6']
        fluxerr_5 = gaia_id_data['fluxerr_6']
        Tmag = gaia_id_data['Tmag'][0]

        # Exclude stars with flux > 230000 counts
        if np.max(flux_3) > 250000:
            print('Stars with gaia_id = {} and Tmag = {:.2f} have been excluded'.format(gaia_id, Tmag))
            continue

        print('The star with gaia_id = {} and Tmag = {:.2f} is used'.format(gaia_id, Tmag))
        num_stars_used += 1

        print('Total number of stars used: ', num_stars_used)
        trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), flux_3, 2), jd_mid - int(jd_mid[0]))
        dt_flux = flux_3 / trend
        dt_fluxerr = fluxerr_5 / trend
        RMS_values = []
        time_seconds = []
        for i in range(1, max_binning):
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, dt_flux, dt_fluxerr, i)
            exposure_time_seconds = i * 10  # 10 seconds per binning
            RMS = np.std(dt_flux_binned)
            RMS_values.append(RMS)
            time_seconds.append(exposure_time_seconds)

        average_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

        # Stop if the number of stars used reaches the specified number
        if num_stars_used >= num_stars:
            break

    # Calculate the average RMS across all stars for each bin
    average_rms_values = np.mean(average_rms_values, axis=0)
    print(average_rms_values)

    # average_rms_values = 10e6 * average_rms_values # Convert to ppm

    # Generate binning times
    binning_times = [i for i in range(1, max_binning)]

    # Calculate the expected decrease in RMS
    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    # Plot RMS as a function of exposure time along with the expected decrease in RMS
    plt.figure(figsize=(10, 8))
    plt.plot(times_binned[0], average_rms_values, 'o', color='black', label='Actual RMS')
    plt.plot(times_binned[0], RMS_model, '--', color='red', label='Model RMS')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposure time [s]')
    plt.ylabel('RMS')
    plt.title('Average RMS vs Exposure time')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(phot_file):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('--num_stars', type=int, default=5, help='Number of stars to plot')
    args = parser.parse_args()

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Plot the current photometry file
    print(f"Plotting the photometry file {phot_file}...")
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

    # Calculate mean and RMS for the noise model
    plot_rms_time(phot_table, args.num_stars)


def main_loop(phot_files):
    for phot_file in phot_files:
        main(phot_file)


if __name__ == "__main__":
    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Run the main function for each photometry file
    main_loop(phot_files)
