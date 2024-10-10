#!/usr/bin/env python
"""
- Updated to run the script for a given tic_id passed as an argument.
"""
import argparse
import os
import numpy as np
from wotan import flatten
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


def plot_noise_model(comp_mags, comp_rms, tmag):
    data = open_json_file()
    fig, ax = plt.subplots(figsize=(10, 6))
    RMS_list = np.array(data['RMS_list']) / 1e6
    Tmag_list = data['Tmag_list']

    if tmag in Tmag_list:
        index = Tmag_list.index(tmag)
        rms_target = RMS_list[index]
        logger.info(f"RMS for target star with Tmag = {tmag}: {RMS_list[index]:.4f}")
    ax.plot(Tmag_list, RMS_list, 'o', color='c', label='All stars', alpha=0.8)
    ax.plot(comp_mags, comp_rms, 'o', color='b', label='Comparison stars', alpha=0.8)
    ax.plot(tmag, rms_target, 'o', color='r', label='Target star', alpha=0.8)
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS per 10 sec')
    # ax.set_yscale('log')

    dim_mag = max(Tmag_list)
    rms_dim_mag = RMS_list[Tmag_list.index(dim_mag)]
    plt.ylim(rms_dim_mag - 0.01, rms_dim_mag + 0.01)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mags_vs_color(mags, colors, tmag, target_color_index):
    plt.figure(figsize=(10, 6))
    plt.scatter(colors, mags, c='blue', alpha=0.8)
    plt.scatter(target_color_index, tmag, c='red', alpha=0.8)
    plt.xlabel('Gaia BP - RP (Color)')
    plt.ylabel('TESS Magnitude')
    plt.title('Magnitudes vs Gaia BP - RP Color Index')
    plt.grid(True)
    plt.show()


def plot_lightcurves_in_subplots(times, fluxes, fluxerrs, tic_ids):
    n = len(tic_ids)
    cols = 3  # Number of columns for subplots
    max_plots_per_figure = 9  # Maximum number of plots per figure
    plots_per_figure = min(max_plots_per_figure, n)  # Adjust for the number of TIC IDs
    rows_per_figure = (plots_per_figure + cols - 1) // cols  # Number of rows per figure

    num_figures = (n + max_plots_per_figure - 1) // max_plots_per_figure  # Number of figures needed

    # Loop over figures
    for fig_num in range(num_figures):
        fig, axes = plt.subplots(rows_per_figure, cols, figsize=(16, 3 * rows_per_figure),
                                 gridspec_kw={'hspace': 0.5, 'wspace': 0.2},
                                 squeeze=False)

        # Start and end indices for the current figure
        start_idx = fig_num * max_plots_per_figure
        end_idx = min(start_idx + max_plots_per_figure, n)

        # Loop over light curves for the current figure
        for i in range(start_idx, end_idx):
            time = times[i]
            flux = fluxes[i]
            fluxerr = fluxerrs[i]
            tic_id = tic_ids[i]

            row = (i - start_idx) // cols
            col = (i - start_idx) % cols

            ax = axes[row, col]
            time_binned, flux_binned, fluxerr_binned = bin_time_flux_error(time, flux, fluxerr, 12)
            ax.errorbar(time_binned, flux_binned, yerr=fluxerr_binned, fmt='o', color='blue', alpha=0.7)
            ax.set_xlabel('Time (JD)', fontsize=10)
            ax.set_ylabel('Flux', fontsize=10)
            ax.set_title(f'Light Curve for TIC ID {tic_id}', fontsize=10)
            ax.grid(True)

            # Make the ticks smaller
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)

        # Hide unused subplots in the last figure (if applicable)
        for j in range(end_idx - start_idx, rows_per_figure * cols):
            fig.delaxes(axes[j // cols, j % cols])

        # Adjust the space around and between subplots using subplots_adjust
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.4, wspace=0.2)
        plt.show()


def relative_phot(table, tic_id_to_plot, bin_size, APERTURE, EXPOSURE, comp_stars_txt=None):
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

    # check which stars are these
    valid_color_data_tic_ids = np.unique(valid_color_data['tic_id'])
    logger.info(f"Total number of stars with valid color information: {len(valid_color_data_tic_ids)}")

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
    # time_clipped, fluxes_clipped, fluxerrs_clipped, airmass_clipped, zero_point_clipped = (
    #     remove_outliers(jd_mid_star, fluxes_star, fluxerrs_star, air_mass=airmass_list, zero_point=zero_point_list)
    # )
    time_clipped, fluxes_clipped, fluxerrs_clipped, airmass_clipped, zero_point_clipped = (jd_mid_star, fluxes_star,
                                                                                           fluxerrs_star,
                                                                                           airmass_list,
                                                                                           zero_point_list)

    avg_zero_point = np.mean(zero_point_clipped)
    avg_magnitude = -2.5 * np.log10(np.mean(fluxes_clipped) / EXPOSURE) + avg_zero_point
    logger.info(f"The target star has TIC ID = {tic_id_to_plot}, TESS magnitude = {tmag:.2f}, "
                f"and calculated magnitude = {avg_magnitude:.2f}")

    # Calculate the color index for all stars
    color_index = valid_color_data['gaiabp'] - valid_color_data['gaiarp']

    color_tolerance = 0.1
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

    if comp_stars_txt is not None:
        # Read the comparison stars from the text file
        with open(comp_stars_txt, 'r') as f:
            comp_stars = f.readlines()
            # take the first column without the header
            tic_ids = np.array([int(line.split()[0]) for line in comp_stars if line[0].isdigit()])
            logger.info(f"Found comparison stars using the comparison text file: {tic_ids}")

    else:
        tic_ids = np.unique(master_star_data['tic_id'])

    rms_comp_list = []
    comparison_fluxes = []
    comparison_fluxerrs = []
    comparison_fluxes_dt = []
    comparison_fluxerrs_dt = []
    comparison_times = []

    for tic_id in tic_ids:
        comp_fluxes = master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
        comp_fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id][f'fluxerr_{APERTURE}']
        comp_time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
        # time_stars, fluxes_stars, fluxerrs_stars, _, _ = remove_outliers(time, fluxes, fluxerrs)
        time_stars, fluxes_stars, fluxerrs_stars = comp_time, comp_fluxes, comp_fluxerrs

        trend, fluxes_dt_comp, fluxerrs_dt_comp = calculate_trend_and_flux(time_stars, fluxes_stars, fluxerrs_stars)
        rms = np.std(fluxes_dt_comp)
        rms_comp_list.append(rms)

        # Collect data for plotting light curves
        comparison_times.append(time_stars)
        comparison_fluxes.append(fluxes_stars)
        comparison_fluxerrs.append(fluxerrs_stars)
        comparison_fluxes_dt.append(fluxes_dt_comp)
        comparison_fluxerrs_dt.append(fluxerrs_dt_comp)

    rms_comp_array = np.array(rms_comp_list)
    min_rms_index = np.argmin(rms_comp_array)
    min_rms_tic_id = tic_ids[min_rms_index]
    min_rms_value = rms_comp_array[min_rms_index]

    logger.info(f"Min Comp star with min RMS: TIC ID = {min_rms_tic_id}, RMS = {min_rms_value:.4f}")

    threshold = SIGMA * min_rms_value
    logger.info(f"Threshold for {SIGMA}-sigma clipping: {threshold:.4f}")

    if comp_stars_txt is not None:
        filtered_tic_ids = tic_ids
    else:
        filtered_tic_ids = tic_ids[rms_comp_array < threshold]
    logger.info(f"Comp stars after filtering by sigma clipping: {len(filtered_tic_ids)}")

    reference_fluxes = np.sum([master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
                               for tic_id in filtered_tic_ids], axis=0)

    plt.plot(time_clipped, reference_fluxes, 'o', label='Reference Star', color='blue')
    plt.show()

    plt.plot(time_clipped, fluxes_clipped, 'o', label='Target Star', color='red')
    plt.show()

    for tic_id in filtered_tic_ids:
        # divide each tic_id with the reference flux and plot them
        comp_fluxes = master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
        comp_dt_fluxes = comp_fluxes / reference_fluxes
        comp_rms = np.std(comp_dt_fluxes)
        # use plot_lightcurves_in_subplots function to plot all the comparison stars
        plt.plot(time_clipped, comp_dt_fluxes, 'o', label=f'RMS = {comp_rms}', alpha=0.8)
        plt.title(f'Comp star: {tic_id}')
        plt.xlabel(f'BJD time')
        plt.ylabel(f'Flux ratio')
        plt.legend(loc='best')
        plt.show()

    # Calculate the flux ratio for the target star with respect the summation of the reference stars fluxes
    flux_ratio = fluxes_clipped / reference_fluxes
    # Calculate the average flux ratio of the target star
    flux_ratio_mean = np.mean(flux_ratio)
    # Normalize the flux ratio (result around unity)
    dt_flux = flux_ratio / flux_ratio_mean
    dt_fluxerr = dt_flux * np.sqrt(
        (fluxerrs_clipped / fluxes_clipped) ** 2 + (fluxerrs_clipped[0] / fluxes_clipped[0]) ** 2)

    plot_lc(dt_flux, time_clipped, np.std(dt_flux), tic_id_to_plot, target_tmag)
    # trend, dt_flux_poly, dt_fluxerr_poly = calculate_trend_and_flux(time_clipped, dt_flux, dt_fluxerr)
    trend, dt_flux_poly, dt_fluxerr_poly = time_clipped, dt_flux, dt_fluxerr

    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux_poly,
                                                                         dt_fluxerr_poly, bin_size)

    logger.info(f'The FINAL number of comparison stars is: {len(filtered_tic_ids)}')
    # take mags and rms for these filtered_tic_ids
    comparison_mags_rms = []
    comparison_colors = []

    for tic_id in filtered_tic_ids:
        tic_data = master_star_data[master_star_data['tic_id'] == tic_id]
        comparison_mags_rms.append((tic_data['Tmag'][0], rms_comp_array[tic_ids == tic_id][0]))
        comparison_colors.append(tic_data['gaiabp'][0] - tic_data['gaiarp'][0])

    # split the array to comparison rms and comparison mags
    comparison_mags = np.array([x[0] for x in comparison_mags_rms])
    comparison_rms = np.array([x[1] for x in comparison_mags_rms])

    # Plot the RMS vs magnitudes for all stars
    plot_noise_model(comparison_mags, comparison_rms, tmag)

    # Plot the magnitudes vs color index for all stars
    plot_mags_vs_color(comparison_mags, comparison_colors, target_tmag, target_color_index)

    # Plot light curves for comparison stars
    plot_lightcurves_in_subplots(comparison_times, comparison_fluxes, comparison_fluxerrs, filtered_tic_ids)

    comparison_files_path = os.path.join(os.getcwd(), f'comparison_stars_{tic_id_to_plot}.txt')
    if os.path.exists(comparison_files_path):
        logger.info(f"Comparison stars information already saved to {comparison_files_path}.")
    else:
        # Open the file to write comparison stars information
        with open(f'comparison_stars_{tic_id_to_plot}.txt', 'w') as f:
            # Write the header
            f.write('tic_id\tTmag\tRMS\tColors\tFluxes\tFluxerrs\n')

            # Iterate over filtered TIC IDs, magnitudes, and RMS values
            for tic_id, tmag, rms, colors, fluxes, fluxerrs in zip(filtered_tic_ids, comparison_mags,
                                                                   comparison_rms, comparison_colors,
                                                                   comparison_fluxes, comparison_fluxerrs):
                # Write each row in the specified format
                f.write(f'{tic_id}\t{tmag:.4f}\t{rms:.4f}\t{colors:.4f}\t{fluxes}\t{fluxerrs}\n')

    return (target_tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
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
    parser.add_argument('--comp_stars', type=str, help='Text file containing comparison stars TIC')
    args = parser.parse_args()
    bin_size = args.bin_size
    APERTURE = args.aper
    EXPOSURE = args.exposure
    tic_id_to_plot = args.tic_id
    comp_stars_file = args.comp_stars

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
            result = relative_phot(phot_table, tic_id_to_plot, bin_size, APERTURE, EXPOSURE, comp_stars_file)

            # Check if result is None
            if result is None:
                logger.info(f"Skipping TIC ID {tic_id_to_plot} due to missing color information.")
                continue

            # Unpack the result if it's not None
            (target_tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
             magnitude, airmass_list, zero_point_list) = result

            # Calculate RMS
            rms = np.std(dt_flux_binned)
            logger.info(f"RMS for TIC ID {tic_id_to_plot} = {rms:.4f}")

            plot_lc(dt_flux_binned, time_binned, rms, tic_id_to_plot, target_tmag)

            # Create an Astropy table from the result
            data_list = [(tic_id_to_plot, target_tmag, time_binned, dt_flux_binned, dt_fluxerr_binned,
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
