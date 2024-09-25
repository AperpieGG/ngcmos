#!/usr/bin/env python
"""
- Updated to run the script for a given tic_id passed as an argument.
"""
import argparse
import os
import numpy as np
import logging
from astropy.table import Table
from matplotlib import pyplot as plt
from utils import (plot_images, get_phot_files, read_phot_file, bin_time_flux_error,
                   remove_outliers, extract_phot_file, calculate_trend_and_flux,
                   expand_and_rename_table, open_json_file)

SIGMA = 2

# Set up the logger
logger = logging.getLogger("rel_phot_logger")
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create file handler which logs even debug messages
log_filename = "target_photometry.log"
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)
logger.addHandler(fh)


def plot_noise_model(comp_rms, comp_mags, tmag):
    data = open_json_file()
    fig, ax = plt.subplots(figsize=(10, 8))
    RMS_list = np.array(data['RMS_list']) / 1e6
    tic_ids = data['TIC_IDs']
    mags_list = data['mags_list']
    Tmag_list = data['Tmag_list']

    if tmag in Tmag_list:
        index = Tmag_list.index(tmag)
        rms_target = RMS_list[index]
        logger.info(f"RMS for target star with Tmag = {tmag}: {RMS_list[index]:.4f}")

    ax.plot(Tmag_list, RMS_list, 'o', color='c', label='total stars', alpha=0.8)
    ax.plot(comp_mags, comp_rms, 'o', color='b', label='comparison stars', alpha=0.8)
    ax.plot(tmag, rms_target, 'o', color='r', label='target star', alpha=0.8)
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    # ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_mags_vs_color(mags, colors):
    plt.figure(figsize=(10, 6))
    plt.scatter(colors, mags, c='blue', alpha=0.8)
    plt.xlabel('TESS Magnitude')
    plt.ylabel('Gaia BP - RP (Color)')
    plt.title('Magnitudes vs Gaia BP - RP Color Index')
    plt.grid(True)
    plt.show()


def plot_lightcurves_in_subplots(times, fluxes, fluxerrs, tic_ids):
    n = len(tic_ids)
    cols = 3  # Number of columns for subplots
    rows = (n + cols - 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows),
                             gridspec_kw={'hspace': 0.8, 'wspace': 0.2},  # Adjust space between rows and columns
                             squeeze=False)

    for i, tic_id in enumerate(tic_ids):
        time = times[i]
        flux = fluxes[i]
        fluxerr = fluxerrs[i]
        ax = axes[i // cols, i % cols]
        time_binned, flux_binned, fluxerr_binned = bin_time_flux_error(time, flux, fluxerr, 12)
        ax.errorbar(time_binned, flux_binned, yerr=fluxerr_binned, fmt='o', color='blue', alpha=0.7)
        ax.set_xlabel('Time (JD)', fontsize=10)
        ax.set_ylabel('Flux', fontsize=10)
        ax.set_title(f'Light Curve for TIC ID {tic_id}', fontsize=10)
        ax.grid(True)

        # Make the ticks smaller
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])

    # Adjust the space around and between subplots using subplots_adjust
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.4, wspace=0.2)
    plt.show()


def relative_phot(table, tic_id_to_plot, bin_size, APERTURE, EXPOSURE):
    """
    Create a relative light curve for a specific TIC ID.

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data.
    tic_id_to_plot : int
        TIC ID of the target star to exclude.
    bin_size : int
        Number of images to bin.

    Returns:
        Various outputs related to the relative photometry.
    """
    # Remove rows where either Gaia BP or RP magnitude is missing (NULL values)
    valid_color_data = table[~np.isnan(table['gaiabp']) & ~np.isnan(table['gaiarp'])]

    # Get the Tmag of the target star
    target_star = valid_color_data[valid_color_data['tic_id'] == tic_id_to_plot]

    if len(target_star) == 0:
        logger.error(f"Target star with TIC ID {tic_id_to_plot} has missing color information. Exiting function.")
        return None

    target_tmag = target_star['Tmag'][0]
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]
    logger.info(f'The target has color index = {target_color_index:.2f} and TESS magnitude = {target_tmag:.2f}')

    # Extract data for the target star
    jd_mid_star, tmag, fluxes_star, fluxerrs_star, sky_star = (
        extract_phot_file(table, tic_id_to_plot, aper=APERTURE))
    airmass_list = table[table['tic_id'] == tic_id_to_plot]['airmass']
    zero_point_list = table[table['tic_id'] == tic_id_to_plot]['zp']

    sky_median = np.median(sky_star)
    time_clipped, fluxes_clipped, fluxerrs_clipped, airmass_clipped, zero_point_clipped = (
        remove_outliers(jd_mid_star, fluxes_star, fluxerrs_star, air_mass=airmass_list, zero_point=zero_point_list)
    )

    avg_zero_point = np.mean(zero_point_clipped)
    avg_magnitude = -2.5 * np.log10(np.mean(fluxes_clipped) / EXPOSURE) + avg_zero_point
    logger.info(f"The target star has TIC ID = {tic_id_to_plot}, TESS magnitude = {tmag:.2f}, "
                f"and calculated magnitude = {avg_magnitude:.2f}")

    # Calculate the color index for all stars
    color_index = valid_color_data['gaiabp'] - valid_color_data['gaiarp']

    color_tolerance = 0.2
    magnitude_tolerance = 1

    within_color_limit = valid_color_data[np.abs(color_index - target_color_index) <= color_tolerance]
    logger.info(f'Comp stars within color limit: {len(np.unique(within_color_limit["tic_id"]))}')

    within_magnitude_limit = within_color_limit[np.abs(within_color_limit['Tmag'] - target_tmag)
                                                <= magnitude_tolerance]
    logger.info(f"Comp stars within color and mag limit: {len(np.unique(within_magnitude_limit['tic_id']))}")

    within_magnitude_limit = within_magnitude_limit[within_magnitude_limit['Tmag'] > 9.4]
    logger.info(f"Comp stars dimmer than 9.4 mags: {len(np.unique(within_magnitude_limit['tic_id']))}")

    master_star_data = within_magnitude_limit[within_magnitude_limit['tic_id'] != tic_id_to_plot]
    master_stars_data_tic_ids = np.unique(master_star_data['tic_id'])
    logger.info(f'The comparison stars before RMS filtering are: {master_stars_data_tic_ids}')

    if len(master_stars_data_tic_ids) < 4:
        logger.warning(f"Target TIC ID {tic_id_to_plot} skipped because only {len(master_stars_data_tic_ids)} "
                       f"comparison stars found (less than 5).")
        return None

    tic_ids = np.unique(master_star_data['tic_id'])

    rms_comp_list = []
    comparison_fluxes = []
    comparison_fluxerrs = []
    comparison_times = []

    for tic_id in tic_ids:
        fluxes = master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
        fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id][f'fluxerr_{APERTURE}']
        time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
        time_stars, fluxes_stars, fluxerrs_stars, _, _ = remove_outliers(time, fluxes, fluxerrs)

        # Detrend the light curve and measure rms
        trend, fluxes_dt_comp, fluxerrs_dt_comp = calculate_trend_and_flux(time_stars, fluxes_stars, fluxerrs_stars)
        rms = np.std(fluxes_dt_comp)
        rms_comp_list.append(rms)

        # Collect data for plotting light curves
        comparison_times.append(time_stars)
        comparison_fluxes.append(fluxes_dt_comp)
        comparison_fluxerrs.append(fluxerrs_dt_comp)

    rms_comp_array = np.array(rms_comp_list)
    min_rms_index = np.argmin(rms_comp_array)
    min_rms_tic_id = tic_ids[min_rms_index]
    min_rms_value = rms_comp_array[min_rms_index]

    logger.info(f"Min Comp star with min RMS: TIC ID = {min_rms_tic_id}, RMS = {min_rms_value:.4f}")

    threshold = SIGMA * min_rms_value
    logger.info(f"Threshold for {SIGMA}-sigma clipping: {threshold:.4f}")

    filtered_tic_ids = tic_ids[rms_comp_array < threshold]
    logger.info(f"Comp stars after filtering by sigma clipping: {len(filtered_tic_ids)}")

    filtered_master_star_data = master_star_data[np.isin(master_star_data['tic_id'], filtered_tic_ids)]
    reference_fluxes = np.sum(filtered_master_star_data[f'flux_{APERTURE}'], axis=0)
    reference_flux_mean = np.mean(reference_fluxes)
    logger.info(f"Reference flux mean after filtering: {reference_flux_mean:.2f}")

    flux_ratio = fluxes_clipped / reference_fluxes
    flux_ratio_mean = np.mean(flux_ratio)
    dt_flux = flux_ratio / flux_ratio_mean
    dt_fluxerr = dt_flux * np.sqrt(
        (fluxerrs_clipped / fluxes_clipped) ** 2 + (fluxerrs_clipped[0] / fluxes_clipped[0]) ** 2)

    trend, dt_flux_poly, dt_fluxerr_poly = calculate_trend_and_flux(time_clipped, dt_flux, dt_fluxerr)
    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux_poly,
                                                                         dt_fluxerr_poly, bin_size)

    # Plot comparison stars data
    comp_mags = np.unique(master_star_data['Tmag'])
    comparison_colors = np.unique(master_star_data['gaiabp'] - master_star_data['gaiarp'])

    # Plot the RMS vs magnitudes for all stars
    plot_noise_model(rms_comp_array, comp_mags, tmag)

    # Plot the magnitudes vs color index for all stars
    plot_mags_vs_color(comp_mags, comparison_colors)

    # Plot light curves for comparison stars
    plot_lightcurves_in_subplots(comparison_times, comparison_fluxes, comparison_fluxerrs, filtered_tic_ids)
    # text file and save the comps tic_ids
    comparison_list = Table([filtered_tic_ids], names=['tic_ids'])
    comparison_list.write(f'comparison_{tic_id_to_plot}_stars.txt', format='ascii', overwrite=True)

    return (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
            avg_magnitude, airmass_clipped, zero_point_clipped)


def plot_lc(flux, time, rms, tic_id_to_plot, tmag):
    # Open the FITS file and read the data

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(time, flux, 'o', label=f'RMS = {rms:.4f}', color='red')
    ax1.set_xlabel('Time (BJD)')
    ax1.set_ylabel('Relative Flux')
    ax1.set_ylim(0.95, 1.05)
    ax1.set_title(f'Rel Phot for TIC ID {tic_id_to_plot} and Tmag = {tmag:.2f}')

    ax1.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Perform relative photometry for a given night')
    parser.add_argument('tic_id', type=int, help='TIC ID of the star')
    parser.add_argument('--bin_size', type=int, default=1, help='Number of images to bin')
    parser.add_argument('--aper', type=int, default=4, help='Aperture radius for photometry')
    parser.add_argument('--exposure', type=float, default=10, help='Exposure time for the images')
    args = parser.parse_args()
    bin_size = args.bin_size
    APERTURE = args.aper
    EXPOSURE = args.exposure
    tic_id_to_plot = args.tic_id

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = os.getcwd()

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    logger.info(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        logger.info(f"Photometry file: {phot_file}")

        # Check if the output file already exists
        fits_filename = f"lc_{tic_id_to_plot}_{bin_size}.fits"
        if os.path.exists(fits_filename):
            logger.info(f"Data for {phot_file} already saved to {fits_filename}. Skipping analysis.")
            continue

        # Extract data for the specific TIC ID
        if tic_id_to_plot in np.unique(phot_table['tic_id']):
            logger.info(f"Performing relative photometry for TIC ID = {tic_id_to_plot}")
            # Perform relative photometry
            result = relative_phot(phot_table, tic_id_to_plot, bin_size, APERTURE, EXPOSURE)

            # Check if result is None
            if result is None:
                logger.info(f"Skipping TIC ID {tic_id_to_plot} due to missing color information.")
                continue

            # Unpack the result if it's not None
            (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
             magnitude, airmass_list, zero_point_list) = result

            # Calculate RMS
            rms = np.std(dt_flux_binned)
            logger.info(f"RMS for TIC ID {tic_id_to_plot} = {rms:.4f}")

            plot_lc(dt_flux_binned, time_binned, rms, tic_id_to_plot, tmag)

            # Create an Astropy table from the result
            data_list = [(tic_id_to_plot, tmag, time_binned, dt_flux_binned, dt_fluxerr_binned,
                          rms, sky_median, airmass_list, zero_point_list, magnitude)]
            data_table = Table(rows=data_list, names=('TIC_ID', 'Tmag', 'Time_JD', 'Relative_Flux', 'Relative_Flux_err',
                                                      'RMS', 'Sky', 'Airmass', 'ZP', 'Magnitude'))

            expanded_data_table = expand_and_rename_table(data_table)

            expanded_data_table.write(fits_filename, format='fits', overwrite=True)

            logger.info(f"Data for TIC ID {tic_id_to_plot} saved to {fits_filename}.")
        else:
            logger.info(f"TIC ID {tic_id_to_plot} is not present in the photometry file {phot_file}.")


if __name__ == "__main__":
    main()
