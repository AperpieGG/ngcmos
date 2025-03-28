#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_images, get_phot_files, read_phot_file, bin_time_flux_error, remove_outliers
from wotan import flatten


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('../directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def plot_lc_with_detrend(table, tic_id_to_plot, bin_size, degree, aper):
    """
    Plot the light curve for a specific TIC ID with detrending

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data
    tic_id_to_plot : int
        TIC ID of the star to plot
    bin_size : int
        Number of images to bin
    degree : int
        Degree of polynomial fit for detrending
    aper : int
        Aperture size for photometry

    Returns:
        None

    """
    # Select rows with the specified TIC ID
    tic_id_data = table[table['tic_id'] == tic_id_to_plot]
    # Get jd_mid, flux_2, and fluxerr_2 for the selected rows
    jd_mid = tic_id_data['jd_mid']
    tmag = tic_id_data['Tmag'][0]
    fluxes = tic_id_data[f'flux_{aper}']
    fluxerrs = tic_id_data[f'fluxerr_{aper}']
    time_stars, fluxes_stars, fluxerrs_stars, _, _ = remove_outliers(jd_mid, fluxes, fluxerrs)

    # use polyfit to detrend the light curve
    trend = np.polyval(np.polyfit(time_stars - int(time_stars[0]), fluxes_stars, degree),
                       time_stars - int(time_stars[0]))

    # use wotan to detrend the light curve
    # flatten_flux, trend = flatten(time_stars, fluxes_stars, window_length=0.05,
    # method='mean', return_trend=True, edge_cutoff=0.1)

    dt_flux = fluxes_stars / trend
    dt_fluxerr = fluxerrs_stars / trend

    # Bin the time, flux, and error
    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_stars, dt_flux, dt_fluxerr, bin_size)

    RMS = np.std(dt_flux_binned)
    print(f"RMS for TIC ID {tic_id_to_plot} = {RMS:.4f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot raw flux with wotan model
    ax1.plot(time_stars, fluxes_stars, '.', color='black', label='Raw Flux')
    ax1.plot(time_stars, trend, color='red', label='Model fit')
    ax1.set_title(f'Detrended LC for TIC ID {tic_id_to_plot} (Tmag = {tmag:.2f})')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Flux [e-]')
    ax1.legend()
    ax2.plot(time_stars, dt_flux, '.', color='black', alpha=0.5)
    if bin_size > 1:
        ax2.plot(time_binned, dt_flux_binned, 'o', color='black', markerfacecolor='blue')
    ax2.set_ylabel('Detrended Flux [e-], binned {}'.format(bin_size))
    ax2.set_xlabel('MJD [days]')
    plt.tight_layout()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('tic_id', type=int, help='The TIC ID of the star to plot')
    parser.add_argument('--aper', type=int, default=4, help='Aperture size for photometry')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    parser.add_argument('--degree', type=int, default=2, help='Degree of polynomial fit for detrending')
    args = parser.parse_args()

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = '.'

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        # Check if tic_id exists in the current photometry file
        if args.tic_id in phot_table['tic_id']:
            print('Found star in photometry file:', phot_file)
            plot_lc_with_detrend(phot_table, args.tic_id, args.bin, args.degree, args.aper)
            break  # Stop looping if tic_id is found
        else:
            print(f"TIC ID {args.tic_id} not found in {phot_file}")

    else:
        print(f"TIC ID {args.tic_id} not found in any photometry file.")


if __name__ == "__main__":
    main()
