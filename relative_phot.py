#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import (plot_images, find_current_night_directory,
                   get_phot_files, read_phot_file, bin_time_flux_error, remove_outliers)


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def plot_lc_with_detrend(table, tic_id_to_plot, bin_size):
    """
    Plot the light curve for a specific TIC ID with detrending

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data
    tic_id_to_plot : int
        TIC ID of the star to plot
    bin_size : int
        Number of images to bin

    Returns:
        None

    """
    # Select rows with the specified TIC ID
    tic_id_data = table[table['tic_id'] == tic_id_to_plot]
    # Get jd_mid, flux_2, and fluxerr_2 for the selected rows
    jd_mid = tic_id_data['jd_mid']
    tmag = tic_id_data['Tmag'][0]
    fluxes = tic_id_data['flux_6']
    fluxerrs = tic_id_data['fluxerr_6']

    # Select stars for master reference star
    master_star_data = table[(table['Tmag'] < 11) & (table['Tmag'] > 9.5)]
    master_fluxes = master_star_data['flux_6']

    # Normalize fluxes
    normalized_fluxes = master_fluxes / np.mean(master_fluxes)

    # Calculate mean flux over all selected stars
    master_reference_flux = np.mean(normalized_fluxes, axis=0)

    # Normalize fluxes of the target star
    normalized_fluxes_target = fluxes / np.mean(fluxes)

    # Subtract master reference star from the target star
    detrended_flux = normalized_fluxes_target - master_reference_flux

    # Bin the time and detrended flux
    time_binned, detrended_flux_binned, _ = bin_time_flux_error(jd_mid, detrended_flux, fluxerrs, bin_size)

    RMS = np.std(detrended_flux_binned)
    print(f"RMS for TIC ID {tic_id_to_plot} = {RMS:.4f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot detrended flux
    ax1.plot(jd_mid, detrended_flux, '.', color='black', label='Detrended Flux')
    ax1.set_title(f'Detrended LC for TIC ID {tic_id_to_plot} (Tmag = {tmag:.2f})')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Detrended Flux [Normalized]')
    ax1.legend()
    if bin_size > 1:
        ax2.plot(time_binned, detrended_flux_binned, 'o', color='black', markerfacecolor='blue')
    ax2.set_ylabel(f'Detrended Flux [Normalized], binned {bin_size}')
    ax2.set_xlabel('MJD [days]')
    plt.tight_layout()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('--tic_id', type=int, help='The TIC ID of the star to plot')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        # Check if tic_id exists in the current photometry file
        if args.tic_id in phot_table['tic_id']:
            print('Found star in photometry file:', phot_file)
            plot_lc_with_detrend(phot_table, args.tic_id, args.bin)
            break  # Stop looping if tic_id is found
        else:
            print(f"TIC ID {args.tic_id} not found in {phot_file}")

    else:
        print(f"TIC ID {args.tic_id} not found in any photometry file.")


if __name__ == "__main__":
    main()
