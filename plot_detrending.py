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


def plot_lc_with_detrend(table, gaia_id_to_plot, bin_size):
    """
    Plot the light curve for a specific Gaia ID with detrending

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data
    gaia_id_to_plot : int
        Gaia ID of the star to plot
    bin_size : int
        Number of images to bin

    Returns:
        None

    """
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]
    # Get jd_mid, flux_2, and fluxerr_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    tmag = gaia_id_data['Tmag'][0]

    # Extract fluxes and errors based on Tmag
    if tmag < 11:
        fluxes = gaia_id_data['flux_5']
        fluxerrs = gaia_id_data['fluxerr_5']
    elif 12 > tmag >= 11:
        fluxes = gaia_id_data['flux_4']
        fluxerrs = gaia_id_data['fluxerr_4']
    else:
        fluxes = gaia_id_data['flux_3']
        fluxerrs = gaia_id_data['fluxerr_3']

    # use polyfit to detrend the light curve
    trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), fluxes, 2),
                       jd_mid - int(jd_mid[0]))
    dt_flux = fluxes / trend
    dt_fluxerr = fluxerrs / trend

    # Bin the time, flux, and error
    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, dt_flux, dt_fluxerr, bin_size)

    RMS = np.std(dt_flux_binned)
    print(f"RMS for Gaia ID {gaia_id_to_plot} = {RMS:.2f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot raw flux with wotan model
    ax1.plot(time_binned, fluxes, 'o', color='black', label='Raw Flux')
    ax1.plot(time_binned, trend, color='red', label='Model fit')
    ax1.set_title(f'Detrended LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f})')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Flux [e-]')
    ax1.legend()
    ax2.errorbar(jd_mid, dt_flux_binned, yerr=dt_fluxerr, fmt='o', color='black', label='Detrended Flux')
    ax2.set_ylabel('Detrended Flux [e-]')
    ax2.set_xlabel('MJD [days]')
    ax2.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('--gaia_id', type=int, help='The Gaia ID of the star to plot')
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

        # Check if gaia_id exists in the current photometry file
        if args.gaia_id in phot_table['gaia_id']:
            print('Found star in photometry file:', phot_file)
            plot_lc_with_detrend(phot_table, args.gaia_id, args.bin)
            break  # Stop looping if gaia_id is found
        else:
            print(f"Gaia ID {args.gaia_id} not found in {phot_file}")

    else:
        print(f"Gaia ID {args.gaia_id} not found in any photometry file.")


if __name__ == "__main__":
    main()
