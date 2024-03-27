#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_images, find_current_night_directory, get_phot_files, read_phot_file, bin_time_flux_error


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


def calculate_master_reference_flux(table, min_mag, max_mag):
    """
    Calculate the master reference flux from stars within the specified magnitude range.

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data
    min_mag : float
        Minimum magnitude for selecting reference stars
    max_mag : float
        Maximum magnitude for selecting reference stars

    Returns:
    master_reference_flux : float
        Mean flux of the selected reference stars
    """
    selected_stars = table[(table['Tmag'] >= min_mag) & (table['Tmag'] <= max_mag)]
    print(f"Number of selected stars: {len(selected_stars)}")
    master_reference_flux = np.mean(selected_stars['flux_6'])
    return master_reference_flux


def correct_light_curves(table, master_reference_flux):
    """
    Correct light curves using the master reference flux.

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data
    master_reference_flux : float
        Master reference flux for normalization
    """
    for row in table:
        # Normalize each star's flux by dividing by the master reference flux
        row['flux_4'] /= master_reference_flux
        # Update other flux columns if needed


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

    # Calculate the master reference flux
    min_mag = 10
    max_mag = 12
    master_reference_flux = calculate_master_reference_flux(table, min_mag, max_mag)

    # Normalize each star's flux by dividing by the master reference flux
    dt_flux_binned = fluxes / master_reference_flux

    RMS = np.std(dt_flux_binned)
    print(f"RMS for TIC ID {tic_id_to_plot} = {RMS:.4f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot raw flux with wotan model
    ax1.plot(jd_mid, fluxes, '.', color='black', label='Raw Flux')
    ax1.set_title(f'Detrended LC for TIC-{tic_id_to_plot} (Tmag = {tmag:.2f})')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Flux [e-]')
    ax1.legend()
    ax2.plot(jd_mid, dt_flux_binned, '.', color='black', alpha=0.5)
    ax2.set_xlabel('MJD [days]')
    ax2.set_ylabel('Normalized Flux')
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
            # Correct light curves using the master reference flux
            master_reference_flux = calculate_master_reference_flux(phot_table, 10, 12)
            correct_light_curves(phot_table, master_reference_flux)
            # Plot the light curve with detrending
            plot_lc_with_detrend(phot_table, args.tic_id, args.bin)
            break  # Stop looping if tic_id is found
        else:
            print(f"TIC ID {args.tic_id} not found in {phot_file}")

    else:
        print(f"TIC ID {args.tic_id} not found in any photometry file.")


if __name__ == "__main__":
    main()