#!/usr/bin/env python
"""
Plot light curve for a given TIC ID from a FITS file containing photometry data
The data is taken from the rel_phot_NGFIELD.fits file that is created from relative_phot.py
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from utils import plot_images
import argparse


def plot_lc(filename, tic_id_to_plot, directory):
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

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(time, flux, label=f'RMS = {rms:.4f}')
    ax1.set_xlabel('Time (JD)')
    ax1.set_ylabel('Relative Flux')
    ax1.set_ylim(0.95, 1.05)
    ax1.set_title(f'Rel Phot for TIC ID {tic_id_to_plot} and Tmag = {tmag:.2f}')

    # Create the second x-axis for airmass
    ax2 = ax1.twiny()
    ax2.plot(time, airmass, 'r-')
    ax2.set_xlabel('Airmass')

    ax1.legend()
    plt.tight_layout()
    plt.show()


def main():
    plot_images()
    directory = '.'
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Name of the FITS file containing photometry data')
    parser.add_argument('tic_id', type=int, help='TIC ID of the star')
    args = parser.parse_args()

    plot_lc(args.filename, args.tic_id, directory)


if __name__ == "__main__":
    main()
