#!/usr/bin/env python
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, read_phot_file, bin_time_flux_error


def compute_rms_values(phot_table):
    """Compute RMS values for the provided photometry table for a specific TIC_ID target."""
    # Filter the table for the specific TIC_ID target
    target_tic_id = 269218084
    phot_table = phot_table[phot_table['TIC_ID'] == target_tic_id]

    if len(phot_table) == 0:
        print(f"No data found for TIC_ID: {target_tic_id}")
        return None, None, None
    tmag = phot_table['Tmag']
    print(f'The target has Tmag: {tmag[0]}')
    jd_mid = phot_table['Time_BJD']
    rel_flux = phot_table['Relative_Flux']
    rel_fluxerr = phot_table['Relative_Flux_err']

    print(f"The number of data points for TIC_ID {target_tic_id} is: {len(rel_flux)}")

    # Parameters for binning
    max_binning = 600
    RMS = []
    times_binned = []

    # Compute RMS values for different binning times
    for i in range(1, max_binning):
        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
        RMS.append(np.array(np.std(dt_flux_binned)))
        times_binned.append(i * 10)

    # Define binning times
    binning_values = np.array([i for i in range(1, max_binning)])

    # Compute white model
    white_noise = 1 / np.sqrt(binning_values)
    RMS_model_white = RMS[0] * white_noise ** 2


    return times_binned, RMS, RMS_model_white


def plot_two_rms(times, avg_rms, RMS_model):
    """Generate RMS plot with data and model."""
    plt.figure(figsize=(8, 6))
    plt.plot(times, avg_rms, 'o', color='black', label='Data')
    plt.plot(times, RMS_model, '--', color='black', label='Model')
    plt.axvline(x=900, color='red', linestyle='-', label='Reference Line')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposure time (s)')
    plt.ylabel('RMS (ppm)')
    plt.legend()

    # Format y-axis tick labels
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().tick_params(axis='y', which='minor', length=4)

    plt.tight_layout()
    plt.show()


def process_file():
    """Process a single photometry file."""
    plot_images()  # Optional: Displays images, ensure this function works correctly.
    current_night_directory = '.'
    file_path = os.path.join(current_night_directory, 'rel_phot_NG1858-4651_5_1.fits')
    phot_table = read_phot_file(file_path)
    return phot_table


if __name__ == "__main__":
    # Process the photometry file
    phot_table1 = process_file()

    if phot_table1 is not None:
        # Compute RMS values
        times1, avg_rms1, RMS_model1 = compute_rms_values(phot_table1)

        if times1 is not None and avg_rms1 is not None and RMS_model1 is not None:
            # Plot the results
            plot_two_rms(times1, avg_rms1, RMS_model1)
        else:
            print("Error: RMS computation failed.")
    else:
        print("Error: Failed to process photometry file.")