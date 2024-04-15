#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import (plot_images, find_current_night_directory,
                   get_phot_files, read_phot_file, bin_time_flux_error, remove_outliers, extract_phot_file)


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
    jd_mid, tmag, fluxes, fluxerrs, sky = extract_phot_file(table, tic_id_to_plot)

    time_clipped, fluxes_clipped, fluxerrs_clipped = remove_outliers(jd_mid, fluxes, fluxerrs)

    # Select stars for master reference star, excluding the target star
    master_star_data = table[(table['Tmag'] >= 9) & (table['Tmag'] <= 10.5) & (table['tic_id'] != tic_id_to_plot)]
    print(f"the number of stars with tic_ids are {len(np.unique(master_star_data['tic_id']))}")

    # Check if tic_id_to_plot is included in the master_star_data
    if tic_id_to_plot in np.unique(master_star_data['tic_id']):
        print(f"TIC ID {tic_id_to_plot} is included.")
    else:
        print(f"TIC ID {tic_id_to_plot} is not included.")

    # Calculate reference star flux
    reference_fluxes = np.sum(master_star_data['flux_6'], axis=0)
    reference_flux_mean = np.mean(reference_fluxes)
    print(f"Reference flux mean = {reference_flux_mean:.2f}")

    # Normalize reference star flux
    reference_flux_normalized = reference_fluxes / reference_flux_mean
    print(f"Reference flux normalized = {reference_flux_normalized}")

    # Normalize target star flux
    target_flux_normalized = fluxes_clipped / np.mean(fluxes_clipped)
    print(f"The target flux has tmag = {tmag:.2f}, and tic_id = {tic_id_to_plot}")

    # Perform relative photometry
    dt_flux = target_flux_normalized / reference_flux_normalized
    dt_fluxerr = dt_flux * np.sqrt(
        (fluxerrs_clipped / fluxes_clipped) ** 2 + (fluxerrs_clipped[0] / fluxes_clipped[0]) ** 2)

    # use polynomial to detrend the light curve
    trend = np.polyval(np.polyfit(time_clipped - int(time_clipped[0]), dt_flux, 2),
                       time_clipped - int(time_clipped[0]))
    dt_flux_poly = dt_flux / trend
    dt_fluxerr_poly = dt_fluxerr / trend

    # Bin the time, flux, and error
    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux_poly, dt_fluxerr_poly, bin_size)

    RMS = np.std(dt_flux)
    RMS_binned = np.std(dt_flux_binned)
    print(f"RMS for TIC ID {tic_id_to_plot} = {RMS:.4f}")
    print(f"RMS for TIC ID {tic_id_to_plot} binned = {RMS_binned:.4f}")
    print(f"The tmag is {tmag:.2f}")

    return time_clipped, fluxes_clipped, dt_flux_poly, dt_fluxerr_poly, tmag, time_binned, dt_flux_binned, dt_fluxerr_binned


def plot_relative_lc(time_clipped, fluxes_clipped, dt_flux, dt_fluxerr, tmag, time_binned,
                     dt_flux_binned, tic_id_to_plot, bin_size):
    """
    Plot the relative light curve for a specific TIC ID

    Parameters:
    time_clipped : numpy.ndarray
        Clipped time values
    fluxes_clipped : numpy.ndarray
        Clipped flux values
    trend : numpy.ndarray
        Trend values
    dt_flux : numpy.ndarray
        Detrended flux values
    dt_fluxerr : numpy.ndarray
        Detrended flux error values
    time_binned : numpy.ndarray
        Binned time values
    dt_flux_binned : numpy.ndarray
        Binned detrended flux values
    tic_id_to_plot : int
        TIC ID of the star to plot
    tmag : float
        Tmag value of the star
    bin_size : int
        Number of images to bin

    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot raw flux with wotan model
    ax1.plot(time_clipped, fluxes_clipped, '.', color='black', label='Raw Flux')
    ax1.set_title(f'TIC ID {tic_id_to_plot}, Tmag = {tmag:.2f}, binned {bin_size}')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Relative Flux [e-]')
    ax1.legend()
    ax2.errorbar(time_clipped, dt_flux, yerr=dt_fluxerr, fmt='.', color='black')
    if bin_size > 1:
        ax2.plot(time_binned, dt_flux_binned, 'o', markerfacecolor='red')
        # Set limits only for the binned data axis
        ax2.set_ylim([np.min(dt_flux_binned) * np.abs(np.min(dt_flux_binned)),
                      np.max(dt_flux_binned) * np.abs(np.max(dt_flux_binned))])
    ax2.set_ylabel('Detrended Flux [e-], binned {}'.format(bin_size))
    ax2.set_xlabel('MJD [days]')

    plt.tight_layout()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('tic_id', type=int, help='The TIC ID of the star to plot')
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
            (time_clipped, fluxes_clipped, dt_flux, dt_fluxerr,
             tmag, time_binned, dt_flux_binned, dt_fluxerr_binned) = relative_phot(phot_table, args.tic_id, args.bin)
            plot_relative_lc(time_clipped, fluxes_clipped, dt_flux, dt_fluxerr, tmag, time_binned,
                             dt_flux_binned, args.tic_id, args.bin)
            break  # Stop looping if tic_id is found
        else:
            print(f"TIC ID {args.tic_id} not found in {phot_file}")

    else:
        print(f"TIC ID {args.tic_id} not found in any photometry file.")


if __name__ == "__main__":
    main()
