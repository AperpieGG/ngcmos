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
from astropy.stats import sigma_clip


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


def calculate_mean_rms_flux(table, bin_size, num_stars, directory):
    mean_flux_list = []
    RMS_list = []
    sky_list = []
    tmag_list = []

    for gaia_id in table['gaia_id'][:num_stars]:  # Selecting the first num_stars stars
        gaia_id_data = table[table['gaia_id'] == gaia_id]
        Tmag = gaia_id_data['Tmag'][0]
        jd_mid = gaia_id_data['jd_mid']
        flux_4 = gaia_id_data['flux_6']
        fluxerr_4 = gaia_id_data['fluxerr_6']
        sky_4 = gaia_id_data['flux_w_sky_6'] - gaia_id_data['flux_6']
        skyerrs_4 = np.sqrt(gaia_id_data['fluxerr_6'] ** 2 + gaia_id_data['fluxerr_w_sky_6'] ** 2)

        flux_4_clipped = sigma_clip(flux_4, sigma=3, maxiters=5)

        zp = []
        detrended_mags = []  # List to store detrended magnitudes

        for frame_id in gaia_id_data['frame_id']:
            image_header = fits.getheader(os.path.join(directory, frame_id))
            zp_value = round(image_header['MAGZP_T'], 3)
            zp.append(zp_value)

        # Detrend the flux using zero points
        print('First flux and zp: ', flux_4_clipped[0], zp[0])
        detrended_flux = flux_4_clipped / np.mean(zp)

        # Convert detrended flux to magnitudes using zero points
        detrended_mags = -2.5 * np.log10(detrended_flux) + np.mean(zp)

        # Plot the detrended magnitudes for this star
        plt.plot(jd_mid, detrended_mags, 'o', color='darkgreen', label='Detrended Magnitudes', alpha=0.5)
        plt.xlabel('JD Mid')
        plt.ylabel('Detrended Magnitudes')
        plt.title(f'Detrended Magnitudes for Star {gaia_id}')
        plt.ylim(np.mean(detrended_mags) - 1, np.mean(detrended_mags) + 1)
        plt.show()

        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, flux_4, fluxerr_4, bin_size)

        # Calculate mean flux and RMS
        mean_flux = np.mean(flux_4)
        RMS = np.std(dt_flux_binned) * 1000000
        mean_sky = np.median(sky_4)

        # Append to lists
        mean_flux_list.append(mean_flux)
        RMS_list.append(RMS)
        sky_list.append(mean_sky)
        tmag_list.append(Tmag)

        # # Store gaia_id for stars with RMS lower than 0.005
        # if RMS < 0.005:
        #     low_rms_gaia_ids.append(gaia_id)

    print('The mean RMS is: ', np.mean(RMS_list))
    return mean_flux_list, RMS_list, sky_list, tmag_list


def extract_header(table, image_directory):
    unique_frame_ids = np.unique(table['frame_id'])

    airmass_list = []
    zp_list = []

    for frame_id in unique_frame_ids:
        # Get the path to the FITS file
        fits_file_path = os.path.join(image_directory, frame_id)

        # Read FITS file header to extract airmass
        with fits.open(fits_file_path) as hdul:
            image_header = hdul[0].header
            airmass = round(image_header['AIRMASS'], 3)
            zp = image_header['MAGZP_T']

        # Append airmass value and frame ID to the list
        airmass_list.append(airmass)
        zp_list.append(zp)

    print(f"Average airmass: {np.mean(airmass_list)}")
    print(f"Average ZP: {np.mean(zp)}")

    return airmass_list, zp_list


def scintilation_noise(airmass_list):
    t = 10  # exposure time
    D = 0.2  # telescope diameter
    h = 2433  # height of Paranal
    H = 8000  # height of atmospheric scale
    airmass = np.mean(airmass_list)  # airmass
    C_y = 1.54
    secZ = 1.2  # airmass
    W = 1.75  # wind speed
    # N = 0.09 * (D ** (-2 / 3) * secZ ** W * np.exp(-h / ho)) * (2 * t) ** (-1 / 2)
    N = np.sqrt(10e-6 * (C_y ** 2) * (D ** (-4 / 3)) * (1 / t) * (airmass ** 3) * np.exp((-2. * h) / H))
    print('Scintilation noise: ', N)
    return N


def noise_model(mean_flux_list, RMS_list, tmag_list):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(tmag_list, RMS_list, 'o', color='darkgreen', label='data', alpha=0.5)
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    plt.tight_layout()
    ax.invert_xaxis()
    ax.set_ylim(1000, 100000)
    ax.set_xlim(16, 8)
    plt.legend(loc='best')
    plt.show()


def main(phot_file):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    parser.add_argument('--num_stars', type=int, default=100, help='Number of stars to plot')
    args = parser.parse_args()
    bin_size = args.bin

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Plot the current photometry file
    print(f"Plotting the photometry file {phot_file}...")
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

    # Calculate mean and RMS for the noise model
    mean_flux_list, RMS_list, sky_list, tmag_list = calculate_mean_rms_flux(
        phot_table, bin_size=bin_size, num_stars=args.num_stars, directory=current_night_directory)

    # Plot the noise model
    noise_model(mean_flux_list, RMS_list, tmag_list)


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
