#!/usr/bin/env python
"""
- First, cut the table for stars 9.5-12 mags; these will be used as reference stars
- Exclude the tic_id you want to perform relative photometry (target_flux)
- Measure the RMS for each raw lightcurve for your reference stars
- Find the reference stars with the lowest RMS (2 sigma clipping threshold)
- Use these stars and sum their fluxes (sum_fluxes)
- Find the mean of the reference master flux (mean_ref_flux)
- Normalize the sum_fluxes by dividing by the mean_ref_flux (normalized_reference_flux)
- Normalize the target_flux by dividing by the mean of the target_flux (normalized_target_flux)
- Perform relative photometry by dividing the normalized_target_flux by the normalized_reference_flux (dt_flux)
- Apply a second-order polynomial to correct for color (dt_flux_poly)
"""
import argparse
import os
import numpy as np
import logging
from astropy.table import Table
from utils import (plot_images, get_phot_files, read_phot_file, bin_time_flux_error,
                   remove_outliers, extract_phot_file, calculate_trend_and_flux, expand_and_rename_table)

SIGMA = 2

# Set up the logger
logger = logging.getLogger("rel_phot_logger")
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create file handler which logs even debug messages
log_filename = "relative_photometry.log"
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)
logger.addHandler(fh)


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

    # check which stars are these
    valid_color_data_tic_ids = np.unique(valid_color_data['tic_id'])
    logger.info(f"Total number of stars with valid color information: {len(valid_color_data_tic_ids)}")

    # Get the Tmag of the target star
    target_star = valid_color_data[valid_color_data['tic_id'] == tic_id_to_plot]

    # Check if the target star has valid color data
    if len(target_star) == 0:
        logger.error(f"Target star with TIC ID {tic_id_to_plot} has missing color information. Exiting function.")
        return None

    target_tmag = target_star['Tmag'][0]

    # Calculate the color index of the target star
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]

    logger.info(f"Target star (TIC ID {tic_id_to_plot}) color index (Gaia BP - RP): {target_color_index:.2f}")

    # Calculate the color index for all stars
    color_index = valid_color_data['gaiabp'] - valid_color_data['gaiarp']

    # Define thresholds
    color_tolerance = 0.2  # Choose stars with a similar color index
    magnitude_tolerance = 1  # Choose stars with similar magnitude

    # Filter the stars to be used as reference stars, exclude the target star
    within_color_limit = valid_color_data[np.abs(color_index - target_color_index) <= color_tolerance]
    print(f'Comp stars within color limit: {len(np.unique(within_color_limit["tic_id"]))}')

    within_magnitude_limit = within_color_limit[np.abs(within_color_limit['Tmag'] - target_tmag)
                                                <= magnitude_tolerance]
    logger.info(f"Comp stars within color and mag limit: {len(np.unique(within_magnitude_limit['tic_id']))}")

    within_magnitude_limit = within_magnitude_limit[within_magnitude_limit['Tmag'] > 9.4]
    logger.info(f"Comp stars dimmer than 9.4 mags: {len(np.unique(within_magnitude_limit['tic_id']))}")

    # Further filter to exclude the target star
    master_star_data = within_magnitude_limit[within_magnitude_limit['tic_id'] != tic_id_to_plot]
    master_stars_data_tic_ids = np.unique(master_star_data['tic_id'])
    logger.info(f"Comparison stars remaining after excluding the target star: {len(master_stars_data_tic_ids)}")

    # Check if there are at least 5 comparison stars
    if len(master_stars_data_tic_ids) < 3:
        logger.warning(f"Target TIC ID {tic_id_to_plot} skipped because only {len(master_stars_data_tic_ids)} "
                       f"comparison stars found (less than 5).")
        return None

    # Extract data for the target star
    jd_mid_star, tmag, fluxes_star, fluxerrs_star, sky_star = (
        extract_phot_file(table, tic_id_to_plot, aper=APERTURE))
    airmass_list = table[table['tic_id'] == tic_id_to_plot]['airmass']
    zero_point_list = table[table['tic_id'] == tic_id_to_plot]['zp']

    # Calculate the median sky value for our star
    sky_median = np.median(sky_star)

    # Remove outliers from the target star
    time_clipped, fluxes_clipped, fluxerrs_clipped, airmass_clipped, zero_point_clipped = (
        remove_outliers(jd_mid_star, fluxes_star, fluxerrs_star, air_mass=airmass_list, zero_point=zero_point_list)
    )

    avg_zero_point = np.mean(zero_point_clipped)
    avg_magnitude = -2.5 * np.log10(np.mean(fluxes_clipped) / EXPOSURE) + avg_zero_point
    logger.info(f"The target star has TIC ID = {tic_id_to_plot}, TESS magnitude = {tmag:.2f}, "
                f"and calculated magnitude = {avg_magnitude:.2f}")

    tic_ids = np.unique(master_star_data['tic_id'])

    rms_comp_list = []

    for tic_id in tic_ids:
        fluxes = master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
        fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id][f'fluxerr_{APERTURE}']
        time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
        bary_time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_bary']
        time_stars, fluxes_stars, fluxerrs_stars, _, _ = remove_outliers(bary_time, fluxes, fluxerrs)

        # Detrend the light curve and measure rms
        trend, fluxes_dt_comp, fluxerrs_dt_comp = calculate_trend_and_flux(time_stars, fluxes_stars, fluxerrs_stars)
        rms = np.std(fluxes_dt_comp)
        rms_comp_list.append(rms)

    rms_comp_array = np.array(rms_comp_list)
    min_rms_index = np.argmin(rms_comp_array)
    min_rms_tic_id = tic_ids[min_rms_index]
    min_rms_value = rms_comp_array[min_rms_index]

    logger.info(f"Number of comparison stars before filtering by RMS: {len(rms_comp_array)}")
    logger.info(f"Comparison star with min RMS: TIC ID = {min_rms_tic_id}, RMS = {min_rms_value:.4f}")

    threshold = SIGMA * min_rms_value
    logger.info(f"Threshold for {SIGMA}-sigma clipping: {threshold:.4f}")

    filtered_tic_ids = tic_ids[rms_comp_array < threshold]
    logger.info(f"Number of comparison stars after filtering by sigma clipping: {len(filtered_tic_ids)}")

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

    return (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
            avg_magnitude, airmass_clipped, zero_point_clipped)


def main():
    parser = argparse.ArgumentParser(description='Perform relative photometry for a given night')
    parser.add_argument('--bin_size', type=int, default=1, help='Number of images to bin')
    parser.add_argument('--aper', type=int, default=4, help='Aperture radius for photometry')
    parser.add_argument('--exposure', type=float, default=10, help='Exposure time for the images')
    args = parser.parse_args()
    bin_size = args.bin_size
    APERTURE = args.aper
    EXPOSURE = args.exposure

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
        base_filename = phot_file.split('.')[0]  # Remove the file extension
        fits_filename = f"rel_{base_filename}_{bin_size}.fits"
        if os.path.exists(fits_filename):
            logger.info(f"Data for {phot_file} already saved to {fits_filename}. Skipping analysis.")
            continue

        # Create an empty list to store data for all TIC IDs
        data_list = []

        # Loop through all tic_ids in the photometry file
        for tic_id in np.unique(phot_table['tic_id']):
            # Check if all the Tmag values for the tic_id are less than 14
            if np.all(phot_table['Tmag'][phot_table['tic_id'] == tic_id] <= 12):
                logger.info(f"Performing relative photometry for TIC ID = {tic_id} and with Tmag = "
                            f"{phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]}")
                # Perform relative photometry
                result = relative_phot(phot_table, tic_id, bin_size, APERTURE, EXPOSURE)

                # Check if result is None
                if result is None:
                    logger.info(f"Skipping TIC ID {tic_id} due to missing color information.")
                    continue

                # Unpack the result if it's not None
                (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
                 magnitude, airmass_list, zero_point_list) = result
                
                # Calculate RMS
                rms = np.std(dt_flux_binned)
                logger.info(f"RMS for TIC ID {tic_id} = {rms:.4f}")

                # Append data to the list
                data_list.append((tic_id, tmag, time_binned, dt_flux_binned, dt_fluxerr_binned,
                                  rms, sky_median, airmass_list, zero_point_list, magnitude))
                logger.info('')
            else:
                logger.info(f"TIC ID {tic_id} is not included in the analysis because "
                            f"the Tmag = {phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]} "
                            f"and is dimmer than 12 mags.")
                logger.info('')

        # Create an Astropy table from the data list
        data_table = Table(rows=data_list, names=('TIC_ID', 'Tmag', 'Time_JD', 'Relative_Flux', 'Relative_Flux_err',
                                                  'RMS', 'Sky', 'Airmass', 'ZP', 'Magnitude'))

        expanded_data_table = expand_and_rename_table(data_table)

        expanded_data_table.write(fits_filename, format='fits', overwrite=True)

        logger.info(f"Data for {phot_file} saved to {fits_filename}.")


if __name__ == "__main__":
    main()
