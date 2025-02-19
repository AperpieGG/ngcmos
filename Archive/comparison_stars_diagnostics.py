#!/usr/bin/env python
import sys
# TODO: this has to be archived
from matplotlib import pyplot as plt
import os
import numpy as np
from utils import (plot_images, get_phot_files, read_phot_file,
                   remove_outliers, calculate_trend_and_flux)

SIGMA = 2
APERTURE = 6
EXPOSURE = 10


def plot_comp_stars(table):
    """
    Plot the comparison stars for each photometry file.
    """
    # Select stars for master reference star, excluding the target star
    master_star_data = table[(table['Tmag'] >= 9.5) & (table['Tmag'] <= 12)]
    print(f"Found {len(np.unique(master_star_data['tic_id']))} comparison stars.")

    rms_comp_list = []
    tic_ids = np.unique(master_star_data['tic_id'])

    print(f'The number of comparison stars is {len(tic_ids)}')
    for tic_id in tic_ids:
        fluxes = master_star_data[master_star_data['tic_id'] == tic_id]['flux_6']
        fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id]['fluxerr_6']
        time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
        time_stars, fluxes_stars, fluxerrs_stars, _, _ = remove_outliers(time, fluxes, fluxerrs)

        # Check if the data arrays are empty after outlier removal
        if len(time_stars) == 0 or len(fluxes_stars) == 0 or len(fluxerrs_stars) == 0:
            print(f"No valid data for TIC ID {tic_id} after outlier removal. Skipping.")
            continue

        # Detrend the light curve and measure RMS
        trend, fluxes_dt_comp, fluxerrs_dt_comp = (
            calculate_trend_and_flux(time_stars, fluxes_stars, fluxerrs_stars))
        rms = np.std(fluxes_dt_comp)
        rms_comp_list.append(rms)

    rms_comp_array = np.array(rms_comp_list)
    min_rms_index = np.argmin(rms_comp_array)

    # Get the corresponding TIC ID with the minimum RMS value
    min_rms_value = rms_comp_array[min_rms_index]
    threshold = SIGMA * min_rms_value

    included_tic_ids = [(tic_id, rms) for tic_id, rms in zip(tic_ids, rms_comp_list) if rms < threshold]
    excluded_tic_ids = [(tic_id, rms) for tic_id, rms in zip(tic_ids, rms_comp_list) if rms >= threshold]

    # Prepare data for plotting
    included_mags = [table[table['tic_id'] == tic_id]['Tmag'][0] for tic_id, _ in included_tic_ids]
    excluded_mags = [table[table['tic_id'] == tic_id]['Tmag'][0] for tic_id, _ in excluded_tic_ids]

    included_rms = [rms for _, rms in included_tic_ids]
    excluded_rms = [rms for _, rms in excluded_tic_ids]

    print(f'Included mags and RMS: {len(included_mags)} stars')
    print(f'Excluded mags and RMS: {len(excluded_mags)} stars')

    return included_mags, included_rms, excluded_mags, excluded_rms


def main():
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
        fits_filename = f"comp_{base_filename}.png"
        if os.path.exists(fits_filename):
            print(f"Data for {phot_file} already saved to {fits_filename}. Skipping analysis.")
            continue

        # Plot comparison stars for the current photometry file
        included_mags, included_rms, excluded_mags, excluded_rms = plot_comp_stars(phot_table)

        # Determine y-limits based on RMS values
        min_rms = min(included_rms + excluded_rms) if included_rms + excluded_rms else 0
        max_rms = 4 * min_rms

        # Create a plot for the current photometry file
        plt.figure(figsize=(10, 6))
        plt.scatter(included_mags, included_rms, label=f'{len(included_mags)} Included Stars', color='black', s=50)
        plt.scatter(excluded_mags, excluded_rms, label=f'{len(excluded_mags)} Excluded Stars', color='red', s=50)
        plt.xlabel('Magnitude (Tmag)')
        plt.ylabel('RMS')
        plt.ylim(min_rms - 0.002, max_rms + 0.002)
        plt.title(f'RMS vs Magnitude for Comparison Stars - {phot_file}')
        plt.legend()
        plt.grid(True)

        plt.savefig(fits_filename)
        plt.show()

        print(f"Plot saved to {fits_filename}.")

    print("All photometry files processed.")


if __name__ == "__main__":
    main()
