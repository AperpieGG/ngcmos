#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from utils import plot_images, find_current_night_directory, get_phot_files, read_phot_file, bin_time_flux_error
from astropy.stats import sigma_clip


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('../directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def calculate_mean_rms_flux(table, bin_size, num_stars):
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

        # # exclude stars with flux > 200000
        # if np.max(flux_4) > 200000:
        #     print('Stars with gaia_id = {} and Tmag = {:.2f} have been excluded'.format(gaia_id, Tmag))
        #     continue

        # Sigma clipping
        clipped_flux = sigma_clip(flux_4, sigma=4, maxiters=5)

        # Mask outliers in flux_4 and jd_mid
        flux_4_clipped = flux_4[~clipped_flux.mask]
        fluxerr_4_clipped = fluxerr_4[~clipped_flux.mask]
        jd_mid_clipped = jd_mid[~clipped_flux.mask]

        trend = np.polyval(np.polyfit(jd_mid_clipped - int(jd_mid_clipped[0]), flux_4_clipped, 2), jd_mid_clipped - int(jd_mid_clipped[0]))
        dt_flux = flux_4_clipped / trend
        dt_fluxerr = fluxerr_4_clipped / trend

        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid_clipped, dt_flux, dt_fluxerr, bin_size)

        # Calculate mean flux and RMS
        mean_flux = np.mean(flux_4_clipped)
        RMS = np.std(dt_flux_binned) * 1000000  # Convert to ppm
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
    num_clipped = len(flux_4) - len(flux_4_clipped)
    print('Number of data points clipped:', num_clipped)
    # print('Gaia IDs with RMS < 0.005:', low_rms_gaia_ids)  # Print the array of gaia_id values for low RMS stars

    return mean_flux_list, RMS_list, sky_list


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
    trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), flux_2, 2), jd_mid - int(jd_mid[0]))

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


def noise_sources(sky_list, bin_size, airmass_list):
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

    synthetic_flux = np.arange(100, 1e7, 10)

    # set dark current rate from cmos characterisation
    dark_current_rate = 1.6
    dark_current = dark_current_rate * exposure_time * npix
    dc_noise = np.sqrt(dark_current) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm

    # set read noise from cmos characterisation
    read_noise_pix = 1.56
    read_noise = (read_noise_pix * np.sqrt(npix)) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm
    read_signal = npix * (read_noise_pix ** 2)

    # set random sky background
    sky_flux = np.mean(sky_list)
    sky_noise = np.sqrt(sky_flux) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm
    print('Average sky flux: ', sky_flux)

    # set random photon shot noise from the flux
    photon_shot_noise = np.sqrt(synthetic_flux) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm

    N = scintilation_noise(airmass_list)

    N_sc = (N * synthetic_flux) ** 2
    N = N / np.sqrt(bin_size) * 1000000  # Convert to ppm

    total_noise = np.sqrt(synthetic_flux + sky_flux + dark_current + read_signal + N_sc)
    RNS = total_noise / synthetic_flux / np.sqrt(bin_size)
    RNS = RNS * 1000000  # Convert to ppm

    return synthetic_flux, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS


def noise_model(synthetic_flux, photon_shot_noise, sky_noise, read_noise, dc_noise,
                mean_flux_list, RMS_list, N, RNS):

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(mean_flux_list, RMS_list, 'o', color='darkgreen', label='Noise Model', alpha=0.5)

    ax.plot(synthetic_flux, RNS, color='black', label='total noise')
    ax.plot(synthetic_flux, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(synthetic_flux, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(synthetic_flux, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(synthetic_flux, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(synthetic_flux, np.ones(len(synthetic_flux)) * N, color='orange', label='scintilation noise',
            linestyle='--')
    ax.set_xlabel('Flux [e-]')
    ax.set_ylabel('RMS')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1000, 100000)
    ax.set_xlim(100, 1e6)
    plt.tight_layout()

    plt.legend(loc='best')
    plt.show()


def main(phot_file):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('--gaia_id', type=int, help='The Gaia ID of the star to plot')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    parser.add_argument('--num_stars', type=int, default=100, help='Number of stars to plot')
    args = parser.parse_args()
    gaia_id_to_plot = args.gaia_id
    bin_size = args.bin

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Plot the current photometry file
    print(f"Plotting the photometry file {phot_file}...")
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

    if gaia_id_to_plot:
        plot_lc_with_detrend(phot_table, gaia_id_to_plot)
    else:
        # Calculate mean and RMS for the noise model
        mean_flux_list, RMS_list, sky_list = calculate_mean_rms_flux(phot_table, bin_size=bin_size,
                                                                     num_stars=args.num_stars)

        # Extract airmass from the photometry table
        airmass_list, zp = extract_header(phot_table, current_night_directory)

        # Calculate noise sources
        (synthetic_flux, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS) \
            = noise_sources(sky_list, bin_size, airmass_list)

        # Plot the noise model
        noise_model(synthetic_flux, photon_shot_noise, sky_noise, read_noise, dc_noise, mean_flux_list, RMS_list, N, RNS)


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
