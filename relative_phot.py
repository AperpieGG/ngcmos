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

    time_clipped, fluxes_clipped, fluxerrs_clipped = remove_outliers(jd_mid, fluxes, fluxerrs)

    # Select stars for master reference star
    master_star_data = table[(table['Tmag'] >= 9) & (table['Tmag'] <= 11)]
    master_fluxes_dict = {}

    # Loop through each unique TIC ID within the specified magnitude range
    for master_tic_id in np.unique(master_star_data['tic_id']):
        # Get the fluxes and corresponding jd_mid for the current star
        star_data = master_star_data[master_star_data['tic_id'] == master_tic_id]
        star_fluxes = star_data['flux_6']
        star_jd_mid = star_data['jd_mid']

        # Add the fluxes of the current star to the dictionary
        for jd, flux in zip(star_jd_mid, star_fluxes):
            if jd not in master_fluxes_dict:
                master_fluxes_dict[jd] = []
            master_fluxes_dict[jd].append(flux)

    # Calculate the average flux for each time point to create the master reference flux
    master_reference_fluxes = []
    for jd in sorted(master_fluxes_dict.keys()):
        average_flux = np.mean(master_fluxes_dict[jd])
        master_reference_fluxes.append(average_flux)

    # Convert master reference fluxes to a numpy array
    master_reference_flux = np.array(master_reference_fluxes)

    fluxes_clipped = fluxes_clipped / master_reference_flux
    fluxerrs_clipped = fluxerrs_clipped / master_reference_flux

    # use polyfit to detrend the light curve
    trend = np.polyval(np.polyfit(time_clipped - int(time_clipped[0]), fluxes_clipped, 2),
                       time_clipped - int(time_clipped[0]))
    dt_flux = fluxes_clipped / trend
    dt_fluxerr = fluxerrs_clipped / trend

    # Bin the time, flux, and error
    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux, dt_fluxerr, bin_size)

    RMS = np.std(dt_flux)
    RMS_binned = np.std(dt_flux_binned)
    print(f"RMS for TIC ID {tic_id_to_plot} = {RMS:.4f}")
    print(f"RMS for TIC ID {tic_id_to_plot} binned = {RMS_binned:.4f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot raw flux with wotan model
    ax1.plot(time_clipped, fluxes_clipped, '.', color='black', label='Raw Flux')
    ax1.plot(time_clipped, trend, color='red', label='Model fit')
    ax1.set_title(f'Detrended LC for TIC ID {tic_id_to_plot} (Tmag = {tmag:.2f})')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Relative Flux [e-]')
    ax1.legend()
    ax2.errorbar(time_clipped, dt_flux, yerr=dt_fluxerr, fmt='.', color='black', alpha=0.2)
    ax2.plot(time_binned, dt_flux_binned, 'o', markerfacecolor='red')
    ax2.set_ylabel('Detrended Flux [e-], binned {}'.format(bin_size))
    ax2.set_xlabel('MJD [days]')

    # Set limits only for the binned data axis
    ax2.set_ylim([np.min(dt_flux_binned) * np.abs(np.min(dt_flux_binned)),
                  np.max(dt_flux_binned) * np.abs(np.max(dt_flux_binned))])
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
