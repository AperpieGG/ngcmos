#!/usr/bin/env python
"""
- First, cut the table for stars 9-11 mags, these will be used as reference stars
- Exclude the tic_id you want to perform relative photometry (target_flux)
- Measure the rms for each raw lightcurve for your reference stars
- Find the references stars with the lowest rms (2 sigma clipping threshold)
- Use this stars and sum their fluxes (sum_fluxes)
- find the mean of the reference master flux (mean_ref_flux)
- Normalize the sum_fluxes by dividing with the mean_ref_flux (normalized_reference_flux)
- Normalize the target_flux by dividing with the mean of the target_flux (normalized_target_flux)
- Perform relative photometry by dividing the normalized_target_flux with the
normalized_reference_flux (dt_flux)
- Apply a second order polynomial to correct from color (dt_flux_poly)
"""
import argparse
import os
import numpy as np
from astropy.table import Table
from utils import (plot_images, get_phot_files, read_phot_file, bin_time_flux_error,
                   remove_outliers, extract_phot_file, calculate_trend_and_flux, extract_airmass_zp)

SIGMA = 2
APERTURE = 6
EXPOSURE = 10


def expand_and_rename_table(phot_table):
    expanded_rows = []

    for row in phot_table:
        jd_mid_values = row['Time_JD']
        relative_flux_values = row['Relative_Flux']
        relative_flux_err_values = row['Relative_Flux_err']
        airmass = row['Airmass']
        zp = row['ZP']

        # Expand jd_mid, relative_flux, and relative_flux_err columns into individual columns
        for i in range(len(jd_mid_values)):
            expanded_row = list(row)
            expanded_row[row.colnames.index('Time_JD')] = jd_mid_values[i]
            expanded_row[row.colnames.index('Relative_Flux')] = relative_flux_values[i]
            expanded_row[row.colnames.index('Relative_Flux_err')] = relative_flux_err_values[i]
            expanded_row[row.colnames.index('Airmass')] = airmass[i]
            expanded_row[row.colnames.index('ZP')] = zp[i]
            expanded_rows.append(expanded_row)

    # Create a new table with expanded columns
    expanded_table = Table(rows=expanded_rows, names=phot_table.colnames)

    return expanded_table


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
        None

    """
    # Select stars for master reference star, excluding the target star
    master_star_data = table[(table['Tmag'] >= 9) & (table['Tmag'] <= 12) &
                             (table['tic_id'] != tic_id_to_plot)]
    print(f"Found {len(np.unique(master_star_data['tic_id']))} "
          f"comparison stars for the target star {tic_id_to_plot}")
    rms_comp_list = []

    jd_mid, tmag, fluxes, fluxerrs, sky = extract_phot_file(table, tic_id_to_plot, aper=APERTURE)
    airmass_list = table[table['tic_id'] == tic_id_to_plot]['airmass']
    zero_point_list = table[table['tic_id'] == tic_id_to_plot]['zp']

    # Calculate the median sky value for our star
    sky_median = np.median(sky)
    # print('The sky median for the TIC ID {} is {}'.format(tic_id_to_plot, sky_median))

    # Remove outliers from the target star
    time_clipped, fluxes_clipped, fluxerrs_clipped, airmass_clipped, zero_point_clipped \
        = remove_outliers(jd_mid, fluxes, fluxerrs, air_mass=airmass_list, zero_point=zero_point_list)

    avg_zero_point = np.mean(zero_point_clipped)
    avg_airmass = np.mean(airmass_clipped)
    avg_magnitude = -2.5 * np.log10(np.mean(fluxes_clipped) / EXPOSURE) + avg_zero_point
    print(f"The target star has TIC ID = {tic_id_to_plot} and TESS magnitude = {tmag:.2f}, "
          f"and magnitude = {avg_magnitude:.2f}")

    for tic_id in np.unique(master_star_data['tic_id']):
        fluxes = master_star_data[master_star_data['tic_id'] == tic_id]['flux_6']
        fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id]['fluxerr_6']
        time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
        time, fluxes, fluxerrs, _, _ = remove_outliers(time, fluxes, fluxerrs)

        # detrend the lc and measure rms
        trend, fluxes_dt_comp, fluxerrs_dt_comp = calculate_trend_and_flux(time, fluxes, fluxerrs)
        # measure rms
        rms = np.std(fluxes_dt_comp)
        rms_comp_list.append(rms)

    min_rms_index = np.argmin(rms_comp_list)
    # Get the corresponding tic_id
    min_rms_tic_id = np.unique(master_star_data['tic_id'])[min_rms_index]
    # Print the tic_id with the minimum rms value
    print(f"Comparison star with min rms is TIC ID = {min_rms_tic_id} "
          f"and RMS = {np.min(rms_comp_list):.4f}")

    # Calculate mean and standard deviation of rms_list
    rms_std = np.std(rms_comp_list)

    # Define the threshold for two sigma clipping
    threshold = SIGMA * rms_std
    print(f"Threshold for 2 sigma clipping = {threshold:.4f}")

    # Get the minimum rms value and its corresponding tic_id
    min_rms_index = np.argmin(rms_comp_list)
    min_rms_value = rms_comp_list[min_rms_index]

    # Filter out comparison stars outside of two sigma clipping from the minimum rms star
    filtered_tic_ids = []
    for tic_id, rms_value in zip(np.unique(master_star_data['tic_id']), rms_comp_list):
        if np.abs(rms_value - min_rms_value) <= threshold:
            filtered_tic_ids.append(tic_id)

    # Print the filtered list of comparison stars
    print("Comparison stars within two sigma clipping from the minimum rms star:")
    for tic_id in filtered_tic_ids:
        print(
            f"TIC ID {tic_id} with RMS = "
            f"{rms_comp_list[np.where(np.unique(master_star_data['tic_id']) == tic_id)[0][0]]:.4f}")
    print(
        f"Number of comp stars within a sigma = "
        f"{len(filtered_tic_ids)} from total of {len(np.unique(master_star_data['tic_id']))}")

    filtered_master_star_data = master_star_data[np.isin(master_star_data['tic_id'], filtered_tic_ids)]

    # Calculate reference star flux using only the filtered comparison stars
    reference_fluxes = np.sum(filtered_master_star_data['flux_6'], axis=0)
    reference_flux_mean = np.mean(reference_fluxes)
    print(f"Reference flux mean = {reference_flux_mean:.2f}")

    # Normalize reference star flux
    # reference_flux_normalized = reference_fluxes / reference_flux_mean
    # print(f"Reference flux normalized = {reference_flux_normalized}")
    #
    # # Normalize target star flux
    # target_flux_normalized = fluxes_clipped / np.mean(fluxes_clipped)

    flux_ratio = fluxes_clipped / reference_fluxes
    flux_ratio_mean = np.mean(flux_ratio)
    # print(f"The target flux has tmag = {tmag:.2f}, and tic_id = {tic_id_to_plot}")

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
    print(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        print(f"Photometry file: {phot_file}")

        # Check if the output file already exists
        base_filename = phot_file.split('.')[0]  # Remove the file extension
        fits_filename = f"rel_{base_filename}_{bin_size}.fits"
        if os.path.exists(fits_filename):
            print(f"Data for {phot_file} already saved to {fits_filename}. Skipping analysis.")
            continue

        # Create an empty list to store data for all TIC IDs
        data_list = []

        # Loop through all tic_ids in the photometry file
        for tic_id in np.unique(phot_table['tic_id']):
            # Check if all the Tmag values for the tic_id are less than 14
            if np.all(phot_table['Tmag'][phot_table['tic_id'] == tic_id] < 14):
                print(f"Performing relative photometry for TIC ID = {tic_id} and with Tmag = "
                      f"{phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]}")
                (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
                 magnitude, airmass_list, zero_point_list) = relative_phot(phot_table, tic_id, args.bin_size)

                # Calculate RMS
                rms = np.std(dt_flux_binned)
                print(f"RMS for TIC ID {tic_id} = {rms:.4f}")

                # Append data to the list
                data_list.append((tic_id, tmag, time_binned, dt_flux_binned, dt_fluxerr_binned,
                                  rms, sky_median, airmass_list, zero_point_list, magnitude))
                print()
            else:
                print(f"TIC ID {tic_id} is not included in the analysis because "
                      f"the Tmag = {phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]} and is greater than 14.")
                print()

        # Create an Astropy table from the data list
        data_table = Table(rows=data_list, names=('TIC_ID', 'Tmag', 'Time_JD', 'Relative_Flux', 'Relative_Flux_err',
                                                  'RMS', 'Sky', 'Airmass', 'ZP', 'Magnitude'))

        expanded_data_table = expand_and_rename_table(data_table)

        expanded_data_table.write(fits_filename, format='fits', overwrite=True)

        print(f"Data for {phot_file} saved to {fits_filename}.")


if __name__ == "__main__":
    main()
