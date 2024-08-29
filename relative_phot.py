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
APERTURE = 6
EXPOSURE = 10

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


def relative_phot(table, tic_id_to_plot, bin_size):
    """
    Create a relative light curve for a specific TIC ID

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data
    tic_id_to_plot : int
        TIC ID of the target star to exclude
    bin_size : int
        Number of images to bin

    Returns:
        Various outputs related to the relative photometry
    """
    # Filter the stars to be used as reference stars, exclude the target star
    target_tmag = table[table['tic_id'] == tic_id_to_plot]['Tmag'][0]  # Get the Tmag of the target star
    master_star_data = table[(table['tic_id'] != tic_id_to_plot) &
                             (np.abs(table['Tmag'] - target_tmag) <= 0.5)]

    logger.info(f"Found {len(np.unique(master_star_data['tic_id']))} "
                f"comparison stars for the target star {tic_id_to_plot}")
    rms_comp_list = []

    jd_mid_star, tmag, fluxes_star, fluxerrs_star, sky_star = (
        extract_phot_file(table, tic_id_to_plot, aper=APERTURE))
    airmass_list = table[table['tic_id'] == tic_id_to_plot]['airmass']
    zero_point_list = table[table['tic_id'] == tic_id_to_plot]['zp']

    # Calculate the median sky value for our star
    sky_median = np.median(sky_star)

    # Remove outliers from the target star
    time_clipped, fluxes_clipped, fluxerrs_clipped, airmass_clipped, zero_point_clipped \
        = remove_outliers(jd_mid_star, fluxes_star, fluxerrs_star,
                          air_mass=airmass_list, zero_point=zero_point_list)

    avg_zero_point = np.mean(zero_point_clipped)
    avg_magnitude = -2.5 * np.log10(np.mean(fluxes_clipped) / EXPOSURE) + avg_zero_point
    logger.info(f"The target star has TIC ID = {tic_id_to_plot} and TESS magnitude = {tmag:.2f}, "
                f"and magnitude = {avg_magnitude:.2f}")

    tic_ids = np.unique(master_star_data['tic_id'])

    for tic_id in tic_ids:
        fluxes = master_star_data[master_star_data['tic_id'] == tic_id]['flux_6']
        fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id]['fluxerr_6']
        time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
        time_stars, fluxes_stars, fluxerrs_stars, _, _ = remove_outliers(time, fluxes, fluxerrs)

        # Detrend the lc and measure rms
        trend, fluxes_dt_comp, fluxerrs_dt_comp = (
            calculate_trend_and_flux(time_stars, fluxes_stars, fluxerrs_stars))
        # Measure rms
        rms = np.std(fluxes_dt_comp)
        rms_comp_list.append(rms)

    # Convert the list to a numpy array for easy manipulation
    rms_comp_array = np.array(rms_comp_list)

    # Find the index of the minimum RMS value
    min_rms_index = np.argmin(rms_comp_array)

    # Get the corresponding TIC ID with the minimum RMS value
    min_rms_tic_id = tic_ids[min_rms_index]
    min_rms_value = rms_comp_array[min_rms_index]
    logger.info(f"The Number of comparison stars before filtering are: {len(rms_comp_array)}")

    # Print the TIC ID with the minimum RMS value
    logger.info(f"Comparison star with min rms is TIC ID = {min_rms_tic_id} and RMS = {min_rms_value:.4f}")

    # Define the threshold for sigma clipping based on the minimum RMS value
    threshold = SIGMA * min_rms_value
    logger.info(f"Threshold for {SIGMA} sigma clipping = {threshold:.4f}")

    # Filter out comparison stars outside the sigma clipping threshold
    filtered_tic_ids = tic_ids[rms_comp_array < threshold]

    logger.info(f"The Number of comparison stars after filtering are: {len(filtered_tic_ids)}")

    # Print the filtered list of comparison stars
    logger.info("Comparison stars within sigma clipping from the minimum RMS star:")
    for tic_id in filtered_tic_ids:
        rms_value = rms_comp_array[tic_ids == tic_id][0]
    logger.info(f"Number of comp stars within sigma = {len(filtered_tic_ids)} from total of {len(tic_ids)}")

    filtered_master_star_data = master_star_data[np.isin(master_star_data['tic_id'], filtered_tic_ids)]

    # Calculate reference star flux using only the filtered comparison stars
    reference_fluxes = np.sum(filtered_master_star_data['flux_6'], axis=0)
    reference_flux_mean = np.mean(reference_fluxes)
    logger.info(f"Reference flux mean = {reference_flux_mean:.2f}")

    # Normalize reference star flux
    flux_ratio = fluxes_clipped / reference_fluxes
    flux_ratio_mean = np.mean(flux_ratio)

    # Perform relative photometry
    dt_flux = flux_ratio / flux_ratio_mean
    dt_fluxerr = dt_flux * np.sqrt(
        (fluxerrs_clipped / fluxes_clipped) ** 2 + (fluxerrs_clipped[0] / fluxes_clipped[0]) ** 2)

    # Correct for color using a second order polynomial
    trend, dt_flux_poly, dt_fluxerr_poly = calculate_trend_and_flux(time_clipped, dt_flux, dt_fluxerr)

    # Bin the time, flux, and error
    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux_poly,
                                                                         dt_fluxerr_poly, bin_size)

    return (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
            avg_magnitude, airmass_clipped, zero_point_clipped)


def main():
    parser = argparse.ArgumentParser(description='Perform relative photometry for a given night')
    parser.add_argument('--bin_size', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()
    bin_size = args.bin_size

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
            if np.all((phot_table['Tmag'][phot_table['tic_id'] == tic_id] >= 9.5) &
                      (phot_table['Tmag'][phot_table['tic_id'] == tic_id] <= 12)):
                logger.info(f"Performing relative photometry for TIC ID = {tic_id} and with Tmag = "
                            f"{phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]}")
                (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
                 magnitude, airmass_list, zero_point_list) = relative_phot(phot_table, tic_id, args.bin_size)

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
                            f"and is outside 9.5-12.")
                logger.info('')

        # Create an Astropy table from the data list
        data_table = Table(rows=data_list, names=('TIC_ID', 'Tmag', 'Time_JD', 'Relative_Flux', 'Relative_Flux_err',
                                                  'RMS', 'Sky', 'Airmass', 'ZP', 'Magnitude'))

        expanded_data_table = expand_and_rename_table(data_table)

        expanded_data_table.write(fits_filename, format='fits', overwrite=True)

        logger.info(f"Data for {phot_file} saved to {fits_filename}.")


if __name__ == "__main__":
    main()


# # Example Gaia data table
# gaia_data = Table.read("gaia_catalog.fits")  # replace with your Gaia data file
#
# # Target star TIC ID and color index
# target_tic_id = 123456789  # replace with your target's TIC ID
# target_star = gaia_data[gaia_data['tic_id'] == target_tic_id]
# target_color_index = target_star['phot_bp_mean_mag'] - target_star['phot_rp_mean_mag']
# target_magnitude = target_star['phot_g_mean_mag']
#
# # Calculate color index for all stars
# color_index = gaia_data['phot_bp_mean_mag'] - gaia_data['phot_rp_mean_mag']
#
# # Define thresholds
# color_tolerance = 0.2  # Choose stars with a similar color index
# magnitude_tolerance = 0.5  # Choose stars with similar magnitude
#
# # Select comparison stars
# comparison_stars = gaia_data[
#     (np.abs(color_index - target_color_index) < color_tolerance) &
#     (np.abs(gaia_data['phot_g_mean_mag'] - target_magnitude) < magnitude_tolerance) &
#     (gaia_data['tic_id'] != target_tic_id)  # Exclude the target star
# ]
#
# # Now, comparison_stars contains only stars similar in color and brightness to your red target star