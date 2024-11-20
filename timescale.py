#! /usr/bin/env python
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, read_phot_file, bin_time_flux_error


def compute_rms_values(phot_table):
    """Compute RMS values for the provided photometry table."""
    phot_table = phot_table[(phot_table['TIC_ID'] >= 269218084)]

    average_rms_values = []
    times_binned = []
    max_binning = 600

    for TIC_ID in phot_table:
        Tmag_data = phot_table[(phot_table['TIC_ID'] >= 269218084)]
        jd_mid = Tmag_data['Time_BJD']
        rel_flux = Tmag_data['Relative_Flux']
        rel_fluxerr = Tmag_data['Relative_Flux_err']
        RMS_data = Tmag_data['RMS']
        print(f'The number of data points are: {len(rel_flux)}')
        RMS_values = []
        time_seconds = []
        for i in range(1, max_binning):
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
            exposure_time_seconds = i * 10
            RMS = np.std(dt_flux_binned)
            RMS_values.append(RMS)
            time_seconds.append(exposure_time_seconds)

        average_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

    average_rms_values = np.median(average_rms_values, axis=0) * 1e6  # Convert to ppm
    times_binned = times_binned[0]  # Use the first time bin set
    print(f'The shape of the rms values is: {average_rms_values.shape}')
    print(f'The times binned is: {times_binned[0]}')

    binning_times = np.array([i for i in range(1, max_binning)])

    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    return times_binned, average_rms_values, RMS_model


def plot_two_rms(times1, avg_rms1, RMS_model1):
    """Generate two RMS plots in a single figure with one row and two columns."""

    plt.plot(times1, avg_rms1, 'o', color='black')
    plt.plot(times1, RMS_model1, '--', color='black')
    plt.axvline(x=900, color='red', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposure time (s)')
    plt.ylabel('RMS (ppm)')

    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().tick_params(axis='y', which='minor', length=4)
    plt.tight_layout()
    plt.show()


def process_file():
    """Process a single photometry file."""
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, 'rel_phot_NG1858-4651_5_1.fits'))
    return phot_table


if __name__ == "__main__":
    # Process both files
    phot_table1 = process_file()

    # Compute RMS values for both files
    times1, avg_rms1, RMS_model1 = compute_rms_values(phot_table1)

    # Plot the results
    plot_two_rms(times1, avg_rms1, RMS_model1)

