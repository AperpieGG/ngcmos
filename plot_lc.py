#!/usr/bin/env python
"""
Plot light curve for a given TIC ID from a FITS file containing photometry data
The data is taken from the rel_phot_NGFIELD.fits file that is created from relative_phot.py
"""
import os

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from utils import plot_images, bin_time_flux_error, get_rel_phot_files, read_phot_file
import argparse


def plot_lc(filename, tic_id_to_plot, bin_size):
    # Open the FITS file and read the data
    with fits.open(filename) as hdul:
        table = hdul[1].data  # Assuming data is in the first extension

    tic_id_data = table[table['TIC_ID'] == tic_id_to_plot]

    if len(tic_id_data) == 0:
        print(f"TIC ID {tic_id_to_plot} not found in the data.")
        return

    tmag = tic_id_data['Tmag'][0]
    time = tic_id_data['Time_JD']
    flux = tic_id_data['Relative_Flux']
    flux_err = tic_id_data['Relative_Flux_err']
    airmass = tic_id_data['Airmass']
    rms = tic_id_data['RMS'][0]

    if bin_size > 1:
        time_binned, flux_binned, flux_err_binned = \
            bin_time_flux_error(time, flux, flux_err, bin_size)
        rms_binned = np.std(flux_binned)
    else:
        time_binned, flux_binned, flux_err_binned, rms_binned\
            = time, flux, flux_err, rms
    fig, ax1 = plt.subplots(figsize=(8, 6))

    if bin_size > 1:
        ax1.plot(time, flux, 'o', color='blue', alpha=0.2)

    ax1.plot(time_binned, flux_binned, 'o', label=f'RMS = {rms_binned:.4f}', color='red')
    ax1.set_xlabel('Time (JD)')
    ax1.set_ylabel('Relative Flux')
    ax1.set_ylim(0.95, 1.05)
    ax1.set_title(f'Rel Phot for TIC ID {tic_id_to_plot} and Tmag = {tmag:.2f}')

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Airmass')

    # Interpolate airmass values at the positions of the primary x-axis ticks
    primary_xticks = ax1.get_xticks()
    interpolated_airmass = np.interp(primary_xticks, time, airmass)
    airmass_ticks = [f'{a:.2f}' for a in interpolated_airmass]

    ax2.set_xticks(primary_xticks)
    ax2.set_xticklabels(airmass_ticks, rotation=45, ha='right')

    ax1.legend()
    plt.tight_layout()
    plt.show()


def main():
    plot_images()
    directory = '.'
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('tic_id', type=int, help='TIC ID of the star')
    parser.add_argument('--bin', type=int, default=1, help='Bin size for time binning')
    args = parser.parse_args()

    filenames = get_rel_phot_files(directory)

    # Loop through photometry files
    for phot_file in filenames:
        phot_table = read_phot_file(os.path.join(directory, phot_file))

        # Check if tic_id exists in the current photometry file
        if args.tic_id in phot_table['tic_id']:
            print('Found star in photometry file:', phot_file)
            plot_lc(phot_file, args.tic_id, args.bin)
            break  # Stop looping if tic_id is found
        else:
            print(f"TIC ID {args.tic_id} not found in {phot_file}")


if __name__ == "__main__":
    main()
