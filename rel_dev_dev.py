#!/usr/bin/env python
"""
- Updated to run the script for a given tic_id passed as an argument.
"""
import argparse
import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt
from utils import (plot_images, get_phot_files, read_phot_file, bin_time_flux_error,
                   extract_phot_file, calculate_trend_and_flux, remove_outliers)

SIGMA = 2
COLOR_TOLERANCE = 0.2
MAGNITUDE_TOLERANCE = 0.5


def target_info(table, tic_id_to_plot):
    target_star = table[table['tic_id'] == tic_id_to_plot]  # Extract the target star data
    target_tmag = target_star['Tmag'][0]  # Extract the TESS magnitude of the target star
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]  # Extract the color index
    # Extract airmass_list from target star = same for every star
    airmass_list = target_star['airmass']
    return target_tmag, target_color_index, airmass_list  # We may have to add fluxes, fluxerrs, sky


def limits_for_comps(table, tic_id_to_plot):
    target_tmag, target_color, airmass_list = target_info(table, tic_id_to_plot)
    color_index = table['gaiabp'] - table['gaiarp']
    color_mask = np.abs(color_index - target_color) <= COLOR_TOLERANCE
    color_data = table[color_mask]
    mag_mask = np.abs(color_data['Tmag'] - target_tmag) <= MAGNITUDE_TOLERANCE
    valid_color_mag_table = color_data[mag_mask]
    valid_color_mag_table = valid_color_mag_table[valid_color_mag_table['Tmag'] > 9.4]
    filtered_table = valid_color_mag_table[valid_color_mag_table['tic_id'] != tic_id_to_plot]

    return filtered_table


tic_id_to_plot = 9725627
current_night_directory = os.getcwd()
phot_file = read_phot_file(current_night_directory)
table = read_phot_file(phot_file)
filtered_table = limits_for_comps(table, tic_id_to_plot)
unique_tic_ids = np.unique(filtered_table['tic_id'])
print(f'The filter table contains {len(unique_tic_ids)} stars')


# def relative_phot(table, tic_id_to_plot, APERTURE, EXPOSURE):
#
#     target_star = table[table['tic_id'] == tic_id_to_plot]  # Extract the target star data
#     target_tmag = target_star['Tmag'][0] # Extract the TESS magnitude of the target star
#     target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]  # Extract the color index of the target star
#
#
#     print(f'The target has color index = {target_color_index:.2f} and TESS magnitude = {target_tmag:.2f}')
#     jd_mid_star, tmag, fluxes_star, fluxerrs_star, sky_star = (
#         extract_phot_file(table, tic_id_to_plot, aper=APERTURE))
#     zero_point_list = table[table['tic_id'] == tic_id_to_plot]['zp']
#     avg_magnitude = -2.5 * np.log10(np.mean(fluxes_star) / EXPOSURE) + np.mean(zero_point_list)
#     print(f"The target star has TIC ID = {tic_id_to_plot}, TESS magnitude = {tmag:.2f}, "
#           f"and calculated magnitude = {avg_magnitude:.2f}")
#
#     # Calculate the color index for all stars
#     color_index = valid_color_data['gaiabp'] - valid_color_data['gaiarp']
#     color_tolerance = 0.2
#     magnitude_tolerance = 1
#     within_color_limit = valid_color_data[np.abs(color_index - target_color_index) <= color_tolerance]
#     print(f'Comp stars within color limit: {len(np.unique(within_color_limit["tic_id"]))}')
#     within_magnitude_limit = within_color_limit[np.abs(within_color_limit['Tmag'] - target_tmag)
#                                                 <= magnitude_tolerance]
#     print(f"Comp stars within color and mag limit: {len(np.unique(within_magnitude_limit['tic_id']))}")
#     within_magnitude_limit = within_magnitude_limit[within_magnitude_limit['Tmag'] > 9.4]
#     print(f"Comp stars dimmer than 9.4 mags: {len(np.unique(within_magnitude_limit['tic_id']))}")
#
#     master_star_data = within_magnitude_limit[within_magnitude_limit['tic_id'] != tic_id_to_plot]
#     master_stars_data_tic_ids = np.unique(master_star_data['tic_id'])
#     print(f'The comparison stars before RMS filtering are: {master_stars_data_tic_ids}')
#     tic_ids = master_stars_data_tic_ids
#
#     # Filter out stars with fluxes outside the range [fluxes_min, fluxes_max]
#     fluxes_min = np.mean(fluxes_star) - 0.5 * np.mean(fluxes_star)
#     fluxes_max = np.mean(fluxes_star) + 0.5 * np.mean(fluxes_star)
#     print(f'Fluxes min = {fluxes_min:.2f}, Fluxes max = {fluxes_max:.2f}')
#
#     # Filter out stars with fluxes outside the range [fluxes_min, fluxes_max]
#     for tic_id in tic_ids:
#         fluxes = master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
#         if np.any(np.mean(fluxes) < fluxes_min) or np.any(np.mean(fluxes) > fluxes_max):
#             master_star_data = master_star_data[master_star_data['tic_id'] != tic_id]
#
#     tic_ids = np.unique(master_star_data['tic_id'])
#     print(f'The comparison stars after filtering are: {tic_ids}')
#     # At this point we have the info for the target star and filtered comparison stars
#     # will a run a loop to find which stars have the lowest RMS using a 2 order polynomial
#     # Loop to find which stars have the lowest RMS using 2nd-order polynomial
#     rms_comp_list = []
#     comparison_fluxes_dt = []
#     comparison_fluxerrs_dt = []
#     comp_fluxes_list = []
#     comp_fluxerrs_list = []
#     comp_time_list = []
#
#     for tic_id in tic_ids:
#         # Get comparison star data for the current TIC ID
#         comp_fluxes = master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
#         comp_fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id][f'fluxerr_{APERTURE}']
#         comp_time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
#         airmass = master_star_data[master_star_data['tic_id'] == tic_id][
#             'airmass']  # Assuming airmass data is available
#         comp_mags0 = master_star_data[master_star_data['tic_id'] == tic_id]['mag']  # Assuming magnitudes are available
#
#         # Use find_bad_comp_stars to filter bad comparison stars based on RMS
#         comp_star_mask, comp_star_rms, iterations = find_bad_comp_stars(comp_fluxes, airmass, comp_mags0)
#
#         # Filter the comparison stars
#         filtered_fluxes = comp_fluxes[comp_star_mask]
#         filtered_fluxerrs = comp_fluxerrs[comp_star_mask]
#
#         # Continue with your existing processing logic
#         trend, fluxes_dt_comp, fluxerrs_dt_comp = calculate_trend_and_flux(comp_time, filtered_fluxes,
#                                                                            filtered_fluxerrs)
#         rms_comp_list.append(np.std(fluxes_dt_comp))
#         comparison_fluxes_dt.append(fluxes_dt_comp)
#         comparison_fluxerrs_dt.append(fluxerrs_dt_comp)
#         comp_time_list.append(comp_time)
#         comp_fluxes_list.append(filtered_fluxes)
#         comp_fluxerrs_list.append(filtered_fluxerrs)
#
#     # Convert lists to arrays for further processing
#     rms_comp_array = np.array(rms_comp_list)
#     min_rms_index = np.argmin(rms_comp_array)
#     min_rms_tic_id = tic_ids[min_rms_index]
#     min_rms_value = rms_comp_array[min_rms_index]
#
#     print(f"Min Comp star with min RMS: TIC ID = {min_rms_tic_id}, RMS = {min_rms_value:.4f}")
#
#     # Get the filtered comparison stars after using find_bad_comp_stars
#     filtered_tic_ids = tic_ids[comp_star_mask]
#
#     print(f"Comp stars after filtering by Ed's clipping: {len(filtered_tic_ids)}")
#
#     # add this point we have found the final comparison stars
#     # we will now plot the raw light curves for the target and comparison stars - all plotting in 2min bins
#     # we need the following plots:
#     # 1. Raw light curves for comparison stars
#     # 2. Raw light curves for target star
#     # 3. Raw light curve of the master comparison star
#     # 4. Raw light curve for each comparison star divided by the master but
#
#     reference_fluxes = np.sum([master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
#                                for tic_id in filtered_tic_ids], axis=0)
#     reference_fluxerrs = np.sqrt(np.sum([master_star_data[master_star_data['tic_id'] == tic_id]
#                                          [f'fluxerr_{APERTURE}'] ** 2
#                                          for tic_id in filtered_tic_ids], axis=0))
#     # plot master raw comparison
#     master_time_binned, master_fluxes_binned, master_fluxerrs_binned = (
#         bin_time_flux_error(comp_time, reference_fluxes, reference_fluxerrs, 12))
#     plt.errorbar(master_time_binned, master_fluxes_binned, yerr=master_fluxerrs_binned, fmt='o', color='green')
#     plt.title(f"Master comparison star")
#     plt.show()
#
#     # plot raw comparison lc
#     # plot_lightcurves_in_subplots(comp_time_list, comp_fluxes_list, comp_fluxerrs_list, filtered_tic_ids)
#     # plt.show()
#
#     # plot raw target lc
#     target_time_binned, target_fluxes_binned, target_fluxerrs_binned = (
#         bin_time_flux_error(jd_mid_star, fluxes_star, fluxerrs_star, 12))
#
#     target_time_binned_no_out, target_fluxes_binned_no_out, target_fluxerrs_binned_no_out, _, _ = (
#         remove_outliers(target_time_binned, target_fluxes_binned, target_fluxerrs_binned))
#
#     plt.errorbar(target_time_binned_no_out, target_fluxes_binned_no_out, yerr=target_fluxerrs_binned_no_out,
#                  fmt='o', color='red')
#     plt.title(f'Target star: {tic_id_to_plot}')
#     # plt.ylim(55500, 61000)  # to exclude the outlier
#     plt.show()
#
#     # plot flattened comparison lc for each tic_id by dividing the master reference flux and
#     # excluding the tic to be plotted
#     plot_lightcurves_in_batches(comp_time_list, comp_fluxes_list, comp_fluxerrs_list, filtered_tic_ids,
#                                 reference_fluxes, reference_fluxerrs, APERTURE)
#
#     # load the fits image on this particular field.
#     image_data = get_image_data(table[table['tic_id'] == tic_id_to_plot]['frame_id'][0])
#
#     # Assuming x, y coordinates are already extracted for the target star and comparison stars
#     # Example: x_target, y_target for the target star
#     x_target = table[table['tic_id'] == tic_id_to_plot]['x'][0]
#     y_target = table[table['tic_id'] == tic_id_to_plot]['y'][0]
#     print(f'Target star coordinates: x = {x_target}, y = {y_target}')
#
#     # Create a circle for the target star (in red)
#     interval = ZScaleInterval()
#     vmin, vmax = np.percentile(image_data, [5, 95])
#     target_circle = plt.Circle((x_target, y_target), radius=5, color='green', fill=False, linewidth=1.5)
#     plt.gca().add_patch(target_circle)
#
#     # Do the same for comparison stars (for example, x_comp, y_comp for each comparison star)
#     for tic_id in filtered_tic_ids:
#         x_comp = table[table['tic_id'] == tic_id]['x'][0]
#         y_comp = table[table['tic_id'] == tic_id]['y'][0]
#         comp_circle = plt.Circle((x_comp, y_comp), radius=5, color='blue', fill=False, linewidth=1.5)
#         plt.gca().add_patch(comp_circle)
#
#         # Add the TIC ID label for each comparison star
#         plt.text(x_comp, y_comp + 10, str(tic_id), color='blue', fontsize=10, ha='center')
#
#     plt.imshow(image_data, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
#     plt.tight_layout()
#     # Add labels and legend
#     plt.xlabel('X Pixel')
#     plt.ylabel('Y Pixel')
#     # plt.legend([target_circle, comp_circle], ['Target', 'Comp Stars'], loc='upper right')
#     plt.title(f'Target: {tic_id_to_plot}')
#     plt.show()
#
#     # finally plot the light curve for the target star flattened by the master
#     # Calculate the flux ratio for the target star with respect the summation of the reference stars fluxes
#     flux_ratio_binned = target_fluxes_binned / master_fluxes_binned
#     flux_ratio = fluxes_star / reference_fluxes
#     # Calculate the average flux ratio of the target star
#     flux_ratio_mean_binned = np.mean(flux_ratio_binned)
#     flux_ratio_mean = np.mean(flux_ratio)
#     # Normalize the flux ratio (result around unity)
#     target_fluxes_dt = flux_ratio_binned / flux_ratio_mean_binned
#     target_fluxes_dt_unbinned = flux_ratio / flux_ratio_mean
#     RMS = np.std(target_fluxes_dt_unbinned)
#
#     plt.plot(target_time_binned, target_fluxes_dt, 'o', color='red', label=f'RMS unbinned = {RMS:.4f}')
#     plt.title(f'Target star: {tic_id_to_plot} divided by master')
#     plt.legend(loc='best')
#     plt.show()
#
#
# def main():
#     parser = argparse.ArgumentParser(description='Perform relative photometry for a given night')
#     parser.add_argument('tic_id', type=int, help='TIC ID of the star')
#     parser.add_argument('--aper', type=int, default=5, help='Aperture radius for photometry, '
#                                                             'CMOS = 5, CCD = 4')
#     parser.add_argument('--exposure', type=float, default=10, help='Exposure time for the images')
#     args = parser.parse_args()
#     APERTURE = args.aper
#     EXPOSURE = args.exposure
#     tic_id_to_plot = args.tic_id
#
#     # Set plot parameters
#     plot_images()
#
#     # Get the current night directory
#     current_night_directory = os.getcwd()
#
#     # Get photometry files with the pattern 'phot_*.fits'
#     phot_files = get_phot_files(current_night_directory)
#
#     # Loop through photometry files
#     for phot_file in phot_files:
#         phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
#
#         print(f"Photometry file: {phot_file}")
#
#         # Extract data for the specific TIC ID
#         if tic_id_to_plot in np.unique(phot_table['tic_id']):
#             print(f"Performing relative photometry for TIC ID = {tic_id_to_plot}")
#             # Perform relative photometry
#             relative_phot(phot_table, tic_id_to_plot, APERTURE, EXPOSURE)
#
#
# if __name__ == "__main__":
#     main()
