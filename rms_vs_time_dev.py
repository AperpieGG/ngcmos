#!/usr/bin/env python

import argparse
import os
import numpy as np
from astropy.table import Table, hstack, vstack
from utils import (plot_images, get_phot_files, read_phot_file, bin_time_flux_error,
                   remove_outliers, extract_phot_file, calculate_trend_and_flux, extract_airmass_zp)

SIGMA = 2
APERTURE = 6
EXPOSURE = 10


def expand_and_stack_table(phot_table):
    expanded_rows = []

    for row in phot_table:
        jd_mid_values = row['jd_mid']
        relative_flux_values = row['relative_flux']

        # Expand jd_mid and relative_flux columns into individual columns
        for i in range(len(jd_mid_values)):
            expanded_row = list(row)
            expanded_row.extend([jd_mid_values[i], relative_flux_values[i]])
            expanded_rows.append(expanded_row)

    # Define new column names
    column_names = list(phot_table.colnames)
    column_names.extend(['jd_mid_expanded', 'relative_flux_expanded'])

    # Create a new table with expanded columns
    expanded_table = Table(rows=expanded_rows, names=column_names)

    return expanded_table


def relative_phot(table, tic_id_to_plot, bin_size):
    # Select stars for master reference star, excluding the target star
    master_star_data = table[(table['Tmag'] >= 9) & (table['Tmag'] <= 12) &
                             (table['tic_id'] != tic_id_to_plot)]
    print(f"Found {len(np.unique(master_star_data['tic_id']))} "
          f"comparison stars for the target star {tic_id_to_plot}")
    rms_comp_list = []

    jd_mid, tmag, fluxes, fluxerrs, sky = extract_phot_file(table, tic_id_to_plot, aper=APERTURE)
    airmass_list = table[table['tic_id'] == tic_id_to_plot]['airmass']
    zero_point_list = table[table['tic_id'] == tic_id_to_plot]['zp']

    sky_median = np.median(sky)
    time_clipped, fluxes_clipped, fluxerrs_clipped, airmass_clipped, zero_point_clipped = remove_outliers(jd_mid,
                                                                                                          fluxes,
                                                                                                          fluxerrs,
                                                                                                          air_mass=airmass_list,
                                                                                                          zero_point=zero_point_list)

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

        trend, fluxes_dt_comp, fluxerrs_dt_comp = calculate_trend_and_flux(time, fluxes, fluxerrs)
        rms = np.std(fluxes_dt_comp)
        rms_comp_list.append(rms)

    min_rms_index = np.argmin(rms_comp_list)
    min_rms_tic_id = np.unique(master_star_data['tic_id'])[min_rms_index]
    print(f"Comparison star with min rms is TIC ID = {min_rms_tic_id} "
          f"and RMS = {np.min(rms_comp_list):.4f}")

    rms_std = np.std(rms_comp_list)
    threshold = SIGMA * rms_std
    print(f"Threshold for 2 sigma clipping = {threshold:.4f}")

    min_rms_index = np.argmin(rms_comp_list)
    min_rms_value = rms_comp_list[min_rms_index]

    filtered_tic_ids = []
    for tic_id, rms_value in zip(np.unique(master_star_data['tic_id']), rms_comp_list):
        if np.abs(rms_value - min_rms_value) <= threshold:
            filtered_tic_ids.append(tic_id)

    print("Comparison stars within two sigma clipping from the minimum rms star:")
    for tic_id in filtered_tic_ids:
        print(
            f"TIC ID {tic_id} with RMS = "
            f"{rms_comp_list[np.where(np.unique(master_star_data['tic_id']) == tic_id)[0][0]]:.4f}")
    print(
        f"Number of comp stars within a sigma = "
        f"{len(filtered_tic_ids)} from total of {len(np.unique(master_star_data['tic_id']))}")

    filtered_master_star_data = master_star_data[np.isin(master_star_data['tic_id'], filtered_tic_ids)]

    reference_fluxes = np.sum(filtered_master_star_data['flux_6'], axis=0)
    reference_flux_mean = np.mean(reference_fluxes)
    print(f"Reference flux mean = {reference_flux_mean:.2f}")

    flux_ratio = fluxes_clipped / reference_fluxes
    flux_ratio_mean = np.mean(flux_ratio)

    dt_flux = flux_ratio / flux_ratio_mean
    dt_fluxerr = dt_flux * np.sqrt(
        (fluxerrs_clipped / fluxes_clipped) ** 2 + (fluxerrs_clipped[0] / fluxes_clipped[0]) ** 2)

    trend, dt_flux_poly, dt_fluxerr_poly = calculate_trend_and_flux(time_clipped, dt_flux, dt_fluxerr)

    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux_poly,
                                                                         dt_fluxerr_poly, bin_size)

    return (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
            avg_magnitude, avg_airmass, avg_zero_point)


def main():
    parser = argparse.ArgumentParser(description='Perform relative photometry for a given night')
    parser.add_argument('--bin_size', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()
    bin_size = args.bin_size

    plot_images()

    current_night_directory = os.getcwd()
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        print(f"Photometry file: {phot_file}")

        base_filename = phot_file.split('.')[0]
        fits_filename = f"rel_{base_filename}_{bin_size}.fits"
        if os.path.exists(fits_filename):
            print(f"Data for {phot_file} already saved to {fits_filename}. Skipping analysis.")
            continue

        data_list = []

        for tic_id in np.unique(phot_table['tic_id']):
            if np.all(phot_table['Tmag'][phot_table['tic_id'] == tic_id] < 14):
                print(f"Performing relative photometry for TIC ID = {tic_id} and with Tmag = "
                      f"{phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]}")
                (tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
                 magnitude, airmass_list, zero_point_list) = relative_phot(phot_table, tic_id, args.bin_size)

                rms = np.std(dt_flux_binned)
                print(f"RMS for TIC ID {tic_id} = {rms:.4f}")

                data_list.append((tic_id, tmag, time_binned, dt_flux_binned, dt_fluxerr_binned,
                                  rms, sky_median, airmass_list, zero_point_list, magnitude))
                print()
            else:
                print(f"TIC ID {tic_id} is not included in the analysis because "
                      f"the Tmag = {phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]} and is greater than 14.")
                print()

        data_table = Table(rows=data_list, names=('TIC_ID', 'Tmag', 'Time_JD', 'Relative_Flux', 'Relative_Flux_err',
                                                  'RMS', 'Sky', 'Airmass', 'ZP', 'Magnitude'))

        expanded_data_table = expand_and_stack_table(data_table)

        expanded_data_table.write(fits_filename, format='fits', overwrite=True)

        print(f"Data for {phot_file} saved to {fits_filename}.")


if __name__ == "__main__":
    main()
