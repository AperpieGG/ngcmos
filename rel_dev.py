#!/usr/bin/env python
"""
- Updated to run the script for a given tic_id passed as an argument.
"""
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import (plot_images, get_phot_files, read_phot_file, bin_time_flux_error,
                   extract_phot_file, calculate_trend_and_flux)

SIGMA = 2


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


def plot_lightcurves_in_batches(time_list, flux_list, fluxerr_list, tic_ids, reference_fluxes, reference_fluxerrs,
                                APERTURE, batch_size=9):
    """
    Plot the light curves for comparison stars in batches of `batch_size` (9 per figure by default).
    """
    total_stars = len(tic_ids)
    num_batches = int(np.ceil(total_stars / batch_size))  # Calculate how many batches we need

    for batch_num in range(num_batches):
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Create 3x3 grid of subplots
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i in range(batch_size):
            idx = batch_num * batch_size + i
            if idx >= total_stars:
                break  # Exit if we exceed the number of stars to plot

            tic_id = tic_ids[idx]
            ax = axes[i // 3, i % 3]  # Select the correct subplot

            comp_fluxes = flux_list[idx]
            comp_fluxerrs = fluxerr_list[idx]
            comp_time = time_list[idx]

            # Subtract the flux of the current comparison star from the master reference flux
            reference_fluxes_comp = reference_fluxes - comp_fluxes

            # Calculate the relative flux and error
            comp_fluxes_dt = comp_fluxes / reference_fluxes_comp
            comp_fluxerrs_dt = np.sqrt(comp_fluxerrs ** 2 + reference_fluxerrs ** 2)

            # Bin the data (optional, can be skipped if not needed)
            comp_time_dt, comp_fluxes_dt_binned, comp_fluxerrs_dt_binned = (
                bin_time_flux_error(comp_time, comp_fluxes_dt, comp_fluxerrs_dt, 12))

            # Plot the light curve in the current subplot
            ax.plot(comp_time_dt, comp_fluxes_dt_binned, 'o', color='blue', alpha=0.8)
            ax.set_title(f'Comparison star: {tic_id}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            
        plt.tight_layout()
        plt.show()


def relative_phot(table, tic_id_to_plot, APERTURE, EXPOSURE):
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
    print(f"Total number of stars with valid color information: {len(np.unique(valid_color_data['tic_id']))}")

    # Get the Tmag of the target star, and extract time, flux, and fluxerr for the target star
    target_star = valid_color_data[valid_color_data['tic_id'] == tic_id_to_plot]
    target_tmag = target_star['Tmag'][0]
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]
    print(f'The target has color index = {target_color_index:.2f} and TESS magnitude = {target_tmag:.2f}')
    jd_mid_star, tmag, fluxes_star, fluxerrs_star, sky_star = (
        extract_phot_file(table, tic_id_to_plot, aper=APERTURE))
    zero_point_list = table[table['tic_id'] == tic_id_to_plot]['zp']
    avg_magnitude = -2.5 * np.log10(np.mean(fluxes_star) / EXPOSURE) + np.mean(zero_point_list)
    print(f"The target star has TIC ID = {tic_id_to_plot}, TESS magnitude = {tmag:.2f}, "
                f"and calculated magnitude = {avg_magnitude:.2f}")

    # Calculate the color index for all stars
    color_index = valid_color_data['gaiabp'] - valid_color_data['gaiarp']
    color_tolerance = 0.1
    magnitude_tolerance = 1
    within_color_limit = valid_color_data[np.abs(color_index - target_color_index) <= color_tolerance]
    print(f'Comp stars within color limit: {len(np.unique(within_color_limit["tic_id"]))}')
    within_magnitude_limit = within_color_limit[np.abs(within_color_limit['Tmag'] - target_tmag)
                                                <= magnitude_tolerance]
    print(f"Comp stars within color and mag limit: {len(np.unique(within_magnitude_limit['tic_id']))}")
    within_magnitude_limit = within_magnitude_limit[within_magnitude_limit['Tmag'] > 9.4]
    print(f"Comp stars dimmer than 9.4 mags: {len(np.unique(within_magnitude_limit['tic_id']))}")

    master_star_data = within_magnitude_limit[within_magnitude_limit['tic_id'] != tic_id_to_plot]
    master_stars_data_tic_ids = np.unique(master_star_data['tic_id'])
    print(f'The comparison stars before RMS filtering are: {master_stars_data_tic_ids}')
    tic_ids = np.unique(master_star_data['tic_id'])

    # At this point we have the info for the target star and filtered comparison stars
    # will a run a loop to find which stars have the lowest RMS using a 2 order polynomial
    rms_comp_list = []
    comparison_fluxes_dt = []
    comparison_fluxerrs_dt = []
    comp_fluxes_list = []
    comp_fluxerrs_list = []
    comp_time_list = []

    for tic_id in tic_ids:
        comp_fluxes = master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
        comp_fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id][f'fluxerr_{APERTURE}']
        comp_time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']

        trend, fluxes_dt_comp, fluxerrs_dt_comp = calculate_trend_and_flux(comp_time, comp_fluxes, comp_fluxerrs)
        rms_comp_list.append(np.std(fluxes_dt_comp))
        comparison_fluxes_dt.append(fluxes_dt_comp)
        comparison_fluxerrs_dt.append(fluxerrs_dt_comp)
        comp_time_list.append(comp_time)
        comp_fluxes_list.append(comp_fluxes)
        comp_fluxerrs_list.append(comp_fluxerrs)

    rms_comp_array = np.array(rms_comp_list)
    min_rms_index = np.argmin(rms_comp_array)
    min_rms_tic_id = tic_ids[min_rms_index]
    min_rms_value = rms_comp_array[min_rms_index]
    print(f"Min Comp star with min RMS: TIC ID = {min_rms_tic_id}, RMS = {min_rms_value:.4f}")
    threshold = SIGMA * min_rms_value
    print(f"Threshold for {SIGMA}-sigma clipping: {threshold:.4f}")
    filtered_tic_ids = tic_ids[rms_comp_array < threshold]
    print(f"Comp stars after filtering by sigma clipping: {len(filtered_tic_ids)}")

    # add this point we have found the final comparison stars
    # we will now plot the raw light curves for the target and comparison stars - all plotting in 2min bins
    # we need the following plots:
    # 1. Raw light curves for comparison stars
    # 2. Raw light curves for target star
    # 3. Raw light curve of the master comparison star
    # 4. Raw light curve for each comparison star divided by the master but

    reference_fluxes = np.sum([master_star_data[master_star_data['tic_id'] == tic_id][f'flux_{APERTURE}']
                               for tic_id in filtered_tic_ids], axis=0)
    reference_fluxerrs = np.sqrt(np.sum([master_star_data[master_star_data['tic_id'] == tic_id]
                                         [f'fluxerr_{APERTURE}'] ** 2
                                        for tic_id in filtered_tic_ids], axis=0))
    # plot master raw comparison
    master_time_binned, master_fluxes_binned, master_fluxerrs_binned = (
        bin_time_flux_error(comp_time, reference_fluxes, reference_fluxerrs, 12))
    plt.errorbar(master_time_binned, master_fluxes_binned, yerr=master_fluxerrs_binned, fmt='o', color='green')
    plt.title(f"Master comparison star")
    plt.show()

    # plot raw comparison lc
    # plot_lightcurves_in_subplots(comp_time_list, comp_fluxes_list, comp_fluxerrs_list, filtered_tic_ids)
    # plt.show()

    # plot raw target lc
    target_time_binned, target_fluxes_binned, target_fluxerrs_binned = (
        bin_time_flux_error(jd_mid_star, fluxes_star, fluxerrs_star, 12))
    plt.errorbar(target_time_binned, target_fluxes_binned, yerr=target_fluxerrs_binned, fmt='o', color='red')
    plt.title(f'Target star: {tic_id_to_plot}')
    plt.ylim(55500, 61000)  # to exclude the outlier
    plt.show()

    # plot flattened comparison lc for each tic_id by dividing the master reference flux and
    # excluding the tic to be plotted
    plot_lightcurves_in_batches(comp_time_list, comp_fluxes_list, comp_fluxerrs_list, filtered_tic_ids,
                                reference_fluxes, reference_fluxerrs, APERTURE)


def main():
    parser = argparse.ArgumentParser(description='Perform relative photometry for a given night')
    parser.add_argument('tic_id', type=int, help='TIC ID of the star')
    parser.add_argument('--aper', type=int, default=4, help='Aperture radius for photometry')
    parser.add_argument('--exposure', type=float, default=10, help='Exposure time for the images')
    args = parser.parse_args()
    APERTURE = args.aper
    EXPOSURE = args.exposure
    tic_id_to_plot = args.tic_id

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = os.getcwd()

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        print(f"Photometry file: {phot_file}")

        # Extract data for the specific TIC ID
        if tic_id_to_plot in np.unique(phot_table['tic_id']):
            print(f"Performing relative photometry for TIC ID = {tic_id_to_plot}")
            # Perform relative photometry
            relative_phot(phot_table, tic_id_to_plot, APERTURE, EXPOSURE)


if __name__ == "__main__":
    main()
