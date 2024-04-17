#!/usr/bin/env python
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
    # Select stars for master reference star, excluding the target star
    master_star_data = table[(table['Tmag'] >= 9) & (table['Tmag'] <= 11) & (table['tic_id'] != tic_id_to_plot)]
    print(f"the number of stars with tic_ids are {len(np.unique(master_star_data['tic_id']))}")
    rms_comp_list = []

    jd_mid, tmag, fluxes, fluxerrs, sky = extract_phot_file(table, tic_id_to_plot)

    print(f"Number of fluxes data points = {len(fluxes)}")

    time_clipped, fluxes_clipped, fluxerrs_clipped = remove_outliers(jd_mid, fluxes, fluxerrs)

    print(f"Number of fluxes data points after clipping = {len(fluxes_clipped)}")

    for tic_id in np.unique(master_star_data['tic_id']):
        fluxes = master_star_data[master_star_data['tic_id'] == tic_id]['flux_6']
        fluxerrs = master_star_data[master_star_data['tic_id'] == tic_id]['fluxerr_6']
        time = master_star_data[master_star_data['tic_id'] == tic_id]['jd_mid']
        time, fluxes, fluxerrs = remove_outliers(time, fluxes, fluxerrs)

        # measure rms
        rms = np.std(fluxes)
        rms_comp_list.append(rms)

        print(f"RMS for TIC ID {tic_id} = {rms:.4f}")
        # make a list of the rms values
    # Find the index of the minimum rms value
    min_rms_index = np.argmin(rms_comp_list)
    # Get the corresponding tic_id
    min_rms_tic_id = np.unique(master_star_data['tic_id'])[min_rms_index]
    # Print the tic_id with the minimum rms value
    print(f"Comparison star with minimum rms is {min_rms_tic_id} with rms value of {np.min(rms_comp_list):.4f}")

    # Calculate mean and standard deviation of rms_list
    rms_std = np.std(rms_comp_list)

    # Define the threshold for two sigma clipping
    threshold = 2 * rms_std
    print(f"Threshold for two sigma clipping = {threshold:.4f}")

    # Get the minimum rms value and its corresponding tic_id
    min_rms_index = np.argmin(rms_comp_list)
    min_rms_value = rms_comp_list[min_rms_index]

    # Filter out comparison stars outside of two sigma clipping from the minimum rms star
    filtered_tic_ids = []
    for tic_id, rms_value in zip(np.unique(master_star_data['tic_id']), rms_comp_list):
        if np.abs(rms_value - min_rms_value) <= threshold:
            filtered_tic_ids.append(tic_id)

    # Print the filtered list of comparison stars
    print("Comparison stars within two sigma clipping from the minimum rms star:")
    for tic_id in filtered_tic_ids:
        print(
            f"TIC ID {tic_id} with RMS = {rms_comp_list[np.where(np.unique(master_star_data['tic_id']) == tic_id)[0][0]]:.4f}")
    print(
        f"Number of comp stars within a sigma = {len(filtered_tic_ids)} from total of {len(np.unique(master_star_data['tic_id']))}")

    # Check if tic_id_to_plot is included in the master_star_data
    if tic_id_to_plot in np.unique(master_star_data['tic_id']):
        print(f"TIC ID {tic_id_to_plot} is included.")
    else:
        print(f"TIC ID {tic_id_to_plot} is not included.")

    filtered_master_star_data = master_star_data[np.isin(master_star_data['tic_id'], filtered_tic_ids)]

    # Calculate reference star flux using only the filtered comparison stars
    reference_fluxes = np.sum(filtered_master_star_data['flux_6'], axis=0)
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
    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux_poly, dt_fluxerr_poly,
                                                                         bin_size)

    RMS = np.std(dt_flux_binned)
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
    exposure_time = 10 / 60  # 10 seconds in minutes
    # Plot raw flux with wotan model
    ax1.plot(time_clipped, fluxes_clipped, '.', color='black', label='Raw Flux')
    ax1.set_title(f'TIC ID {tic_id_to_plot}, Tmag = {tmag:.2f}, binned {bin_size}')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Raw Flux [e-]')
    ax1.legend()
    ax2.errorbar(time_clipped, dt_flux, yerr=dt_fluxerr, fmt='.', color='black')
    if bin_size > 1:
        ax2.plot(time_binned, dt_flux_binned, 'o', markerfacecolor='red')
        # Set limits only for the binned data axis
        ax2.set_ylim([np.min(dt_flux_binned) * np.abs(np.min(dt_flux_binned)),
                      np.max(dt_flux_binned) * np.abs(np.max(dt_flux_binned))])
    ax2.set_ylabel('Relative Flux [e-], binned {:.2f} min'.format(bin_size * exposure_time))
    ax2.set_xlabel('MJD [days]')

    plt.tight_layout()
    plt.show()


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

        # Loop through all tic_ids in the photometry file
        for tic_id in np.unique(phot_table['tic_id']):
            print(f"Performing relative photometry for TIC ID {tic_id}")
            (time_clipped, fluxes_clipped, dt_flux, dt_fluxerr,
             tmag, time_binned, dt_flux_binned, dt_fluxerr_binned) = relative_phot(phot_table, tic_id, bin_size=1)
            plot_relative_lc(time_clipped, fluxes_clipped, dt_flux, dt_fluxerr, tmag, time_binned,
                             dt_flux_binned, tic_id, bin_size=1)


if __name__ == "__main__":
    main()
