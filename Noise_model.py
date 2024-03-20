#!/usr/bin/env python

"""
This script calculates the noise model for the TESS data. The noise is calculated using detrended fluxes from airmass
The fluxes are converted to magnitudes using zero points. The final plot is RMS vs magnitudes.
"""

import argparse
import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.stats import sigma_clip, sigma_clipped_stats
import json
from utils import (find_current_night_directory, read_phot_file, get_phot_files, bin_time_flux_error, plot_images)


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


def calculate_mean_rms_flux(table, bin_size, num_stars, directory):
    mean_flux_list = []
    RMS_list = []
    sky_list = []
    mags_list = []

    for gaia_id in table['gaia_id'][:num_stars]:  # Selecting the first num_stars stars
        gaia_id_data = table[table['gaia_id'] == gaia_id]
        jd_mid = gaia_id_data['jd_mid']
        flux_4 = gaia_id_data['flux_6']
        fluxerr_4 = gaia_id_data['fluxerr_6']
        sky_4 = gaia_id_data['flux_w_sky_6'] - gaia_id_data['flux_6']
        skyerrs_4 = np.sqrt(gaia_id_data['fluxerr_6'] ** 2 + gaia_id_data['fluxerr_w_sky_6'] ** 2)

        print(f"Running light curve for star {gaia_id}...")

        # # Apply sigma clipping to flux and sky arrays
        # flux_mask = np.logical_or(flux_4 <= 0, np.isnan(flux_4))
        # clipped_flux = sigma_clipped_stats(flux_4, sigma=5, mask=flux_mask)
        # flux_4_clipped = clipped_flux
        #
        # sky_mask = np.logical_or(sky_4 <= 0, np.isnan(sky_4))
        # clipped_sky = sigma_clipped_stats(sky_4, sigma=5, mask=sky_mask)
        # sky_4_clipped = clipped_sky

        zp = []
        for frame_id in gaia_id_data['frame_id']:
            image_header = fits.getheader(os.path.join(directory, frame_id))
            zp_value = round(image_header['MAGZP_T'], 3)
            zp.append(zp_value)

        mags = []
        t = 10  # exposure time
        for flux, zp_value in zip(flux_4, zp):
            if flux <= 0 or np.isnan(flux):
                print(f"The Star with negative flux is {gaia_id}")
                # Skip the entire light curve if a flux data point is negative or NaN
                continue
            else:
                # Convert the non-rejected flux value to magnitude using the zero point
                mag = -2.5 * np.log10(flux/t) + zp_value
                mag_error = 1.0857 * fluxerr_4 / flux_4

                mags.append(mag)

        # Plot the magnitudes for this star
        plt.figure(figsize=(10, 4))
        plt.errorbar(jd_mid, mags, yerr=mag_error, fmt='o', color='black')
        plt.xlabel('JD Mid')
        plt.ylabel('Magnitudes')
        plt.title(f'Magnitudes for Star {gaia_id}')
        plt.show()

        # Detrend the flux by converting back to fluxes and normalize by the mean lc
        fluxes_detrended = 10 ** (-0.4 * np.array(mags))  # Convert magnitudes back to fluxes
        mean_flux = np.mean(fluxes_detrended)  # Calculate the average flux
        dt_flux = fluxes_detrended / mean_flux  # Normalize the fluxes by dividing by the average flux

        # trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), flux_4_clipped, 2), jd_mid - int(jd_mid[0]))
        # dt_flux = flux_4_clipped / trend
        dt_fluxerr = fluxerr_4 / mean_flux

        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, dt_flux, dt_fluxerr, bin_size)

        # Calculate mean flux and RMS
        mean_flux = np.mean(flux_4)
        mean_mags = np.mean(mags)
        RMS = np.std(dt_flux_binned)  # Convert to ppm
        mean_sky = np.median(sky_4)

        # Append to lists
        mean_flux_list.append(mean_flux)
        RMS_list.append(RMS)
        sky_list.append(mean_sky)
        mags_list.append(mean_mags)

    return mean_flux_list, RMS_list, sky_list, mags_list, zp


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
    C_y = 1.54  # constant
    N = np.sqrt(10e-6 * (C_y ** 2) * (D ** (-4 / 3)) * (1 / t) * (airmass ** 3) * np.exp((-2. * h) / H))
    print('Scintilation noise: ', N)
    return N


def noise_sources(sky_list, bin_size, airmass_list, zp):
    """
    Returns the noise sources for a given flux

    returns arrays of noise and signal for a given flux

    Parameters
    ----------
    sky_list : list
        values of sky fluxes
    bin_size : int
        number of images to bin
    airmass_list : list
        values of airmass
    zp : list
        values of zero points

    Returns
    -------
    synthetic_mag : array
        values of synthetic magnitudes
    photon_shot_noise : array
        values of photon shot noise
    sky_noise : array
        values of sky noise
    read_noise : array
        values of read noise
    dc_noise : array
        values of dark current noise
    N : array
        values of scintilation noise
    RNS : array
        values of read noise squared
    """

    # set aperture radius
    aperture_radius = 6
    npix = np.pi * aperture_radius ** 2

    # set exposure time and and random flux
    exposure_time = 10

    synthetic_flux = np.arange(100, 1e6, 10)
    synthetic_mag = np.mean(zp) - 2.5*np.log10(synthetic_flux/exposure_time)

    # set dark current rate from cmos characterisation
    dark_current_rate = 1.6
    dark_current = dark_current_rate * exposure_time * npix
    dc_noise = np.sqrt(dark_current) / synthetic_flux / np.sqrt(bin_size)

    # set read noise from cmos characterisation
    read_noise_pix = 1.56
    read_noise = (read_noise_pix * np.sqrt(npix)) / synthetic_flux / np.sqrt(bin_size)
    read_signal = npix * (read_noise_pix ** 2)

    # set random sky background
    sky_flux = np.mean(sky_list)
    sky_noise = np.sqrt(sky_flux) / synthetic_flux / np.sqrt(bin_size)
    print('Average sky flux: ', sky_flux)

    # set random photon shot noise from the flux
    photon_shot_noise = np.sqrt(synthetic_flux) / synthetic_flux / np.sqrt(bin_size)

    N = scintilation_noise(airmass_list)

    N_sc = (N * synthetic_flux) ** 2
    N = N / np.sqrt(bin_size)

    total_noise = np.sqrt(synthetic_flux + sky_flux + dark_current + read_signal + N_sc)
    RNS = total_noise / synthetic_flux / np.sqrt(bin_size)
    RNS = RNS

    return synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS


def noise_model(RMS_list, mags_list, synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS):
    fig, ax = plt.subplots(figsize=(10, 8))
    print('RMS list: ', RMS_list)
    print('mag list: ', mags_list)
    ax.plot(mags_list, RMS_list, 'o', color='darkgreen', label='data', alpha=0.5)

    ax.plot(synthetic_mag, RNS, color='black', label='total noise')
    ax.plot(synthetic_mag, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(synthetic_mag, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(synthetic_mag, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(synthetic_mag, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(synthetic_mag, np.ones(len(synthetic_mag)) * N, color='orange', label='scintilation noise',
            linestyle='--')
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(5.5, 14)
    ax.set_ylim(0.001, 1)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()
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

    airmass_list, zp = extract_header(phot_table, current_night_directory)

    # Calculate mean and RMS for the noise model
    mean_flux_list, RMS_list, sky_list, mags_list, zp = calculate_mean_rms_flux(
        phot_table, bin_size=bin_size, num_stars=args.num_stars, directory=current_night_directory)

    # Get noise sources
    synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS = (
        noise_sources(sky_list, bin_size, airmass_list, zp))

    # Plot the noise model
    noise_model(RMS_list, mags_list, synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS)


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
