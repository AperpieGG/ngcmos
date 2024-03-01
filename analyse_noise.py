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
from sklearn.base import TransformerMixin
import sklearn
from utils import plot_images
from wotan import flatten


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


def calculate_mean_rms_binned(table, bin_size=60, num_stars=1000):
    mean_flux_list = []
    RMS_list = []
    mean_unbinned_list = []
    rms_unbinned_list = []

    for gaia_id in table['gaia_id'][:num_stars]:  # Selecting the first num_stars stars
        gaia_id_data = table[table['gaia_id'] == gaia_id]
        jd_mid = gaia_id_data['jd_mid']
        flux_2 = gaia_id_data['flux_3']
        fluxerr_2 = gaia_id_data['fluxerr_3']

        # # Bin the data
        # jd_mid_binned = [np.mean(jd_mid[i:i + bin_size]) for i in range(0, len(jd_mid), bin_size)]
        # flux_2_binned = [np.mean(flux_2[i:i + bin_size]) for i in range(0, len(flux_2), bin_size)]

        # Use wotan to detrend the light curve detrended_flux, trend = flatten(jd_mid_binned, flux_2_binned,
        # method='mean', window_length=0.01, return_trend=True)
        trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), flux_2, 2), jd_mid - int(jd_mid[0]))
        dt_flux = flux_2 / trend

        # bin the detrdended flux
        dt_flux_binned = [np.mean(dt_flux[i:i + bin_size]) for i in range(0, len(dt_flux), bin_size)]

        # Calculate mean flux and RMS
        mean_flux = np.mean(flux_2)
        RMS = np.std(dt_flux_binned)

        mean_unbinned = np.mean(flux_2)
        rms_unbinned = np.std(dt_flux)

        # Append to lists
        mean_flux_list.append(mean_flux)
        RMS_list.append(RMS)

        mean_unbinned_list.append(mean_unbinned)
        rms_unbinned_list.append(rms_unbinned)

        print(f"Length of binned flux: {len(dt_flux_binned)}")
        print(f"Length of unbinned flux: {len(dt_flux)}")

    return mean_flux_list, RMS_list, mean_unbinned_list, rms_unbinned_list


def plot_noise_model(mean_flux_list, RMS_list, mean_unbinned, rms_unbinned):
    # Plot the noise model
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(mean_flux_list, RMS_list, 'o', color='black', label='Noise Model')
    ax.plot(mean_unbinned, rms_unbinned, 'o', color='red', label='Unbinned')
    ax.set_xlabel('Mean Flux [e-]')
    ax.set_ylabel('RMS [e-]')
    ax.set_title('Noise Model')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_lc_with_detrend(table, gaia_id_to_plot):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]
    # Get jd_mid, flux_2, and fluxerr_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    flux_2 = gaia_id_data['flux_2']
    fluxerr_2 = gaia_id_data['fluxerr_2']
    tmag = gaia_id_data['Tmag'][0]

    # flatten_lc, trend = flatten(jd_mid, flux_2, window_length=0.01, return_trend=True, method='biweight')
    # use polyfit to detrend the light curve
    trend = np.polyval(np.polyfit(jd_mid-int(jd_mid[0]), flux_2, 2), jd_mid-int(jd_mid[0]))

    # Compute Detrended flux and errors
    norm_flux = flux_2 / trend
    relative_err = fluxerr_2 / trend
    rms = np.std(norm_flux)
    print(f"RMS for Gaia ID {gaia_id_to_plot} = {rms:.2f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot raw flux with wotan model
    ax1.plot(jd_mid, flux_2, 'o', color='black', label='Raw Flux 2')
    ax1.plot(jd_mid, trend, color='red', label='Wotan Model')
    ax1.set_title(f'Detrended LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f})')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Flux [e-]')
    ax1.legend()

    ax2.errorbar(jd_mid, norm_flux, yerr=relative_err, fmt='o', color='black', label='Detrended Flux')
    ax2.set_ylabel('Detrended Flux [e-]')
    ax2.set_xlabel('MJD [days]')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('--gaia_id', type=int, help='The Gaia ID of the star to plot')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()
    gaia_id_to_plot = args.gaia_id
    bin_size = args.bin

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Plot the first photometry file
    print(f"Plotting the first photometry file {phot_files[0]}...")
    phot_table = read_phot_file(phot_files[0])

    # Plot the light curve for the specified Gaia ID
    if gaia_id_to_plot:
        plot_lc_with_detrend(phot_table, gaia_id_to_plot)
    else:
        # Calculate mean and RMS for the noise model
        mean_flux_list, RMS_list, mean_unbinned_list, rms_unbinned_list = calculate_mean_rms_binned(phot_table, bin_size=bin_size)
        plot_noise_model(mean_flux_list, RMS_list, mean_unbinned_list, rms_unbinned_list)


if __name__ == "__main__":
    main()