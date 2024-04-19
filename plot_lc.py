#!/usr/bin/env python
"""
Plot light curve for a given TIC ID from a FITS file containing photometry data
The data is taken from the rel_phot_NGFIELD.fits file that is created from relative_phot.py
"""


from astropy.io import fits
import matplotlib.pyplot as plt
from utils import plot_images
import argparse


def search_and_extract_info(filename, tic_id):
    # Load the FITS file containing the relative photometry data
    data_table = fits.getdata(filename)

    # Search for the index of the provided TIC ID in the data table
    index = None
    for i, id in enumerate(data_table['TIC_ID']):
        if id == tic_id:
            index = i
            break

    if index is None:
        print(f"TIC ID {tic_id} not found in the data.")
        return

    # Extract information for the provided TIC ID
    star_time = data_table['Time_JD'][index]  # Time for the star
    star_flux = data_table['Relative_Flux'][index]  # Flux for the star
    tmag = data_table['Tmag'][index]  # Tmag for the star

    # Plot flux versus time for the star
    plt.figure(figsize=(8, 6))
    plt.plot(star_time, star_flux, 'o')
    plt.xlabel('Time (JD)')
    plt.ylabel('Relative Flux (e-)')
    plt.title(f'Relative Photometry for TIC ID {tic_id} (Tmag = {tmag:.2f})')
    plt.ylim(0.95, 1.05)
    plt.show()


def main():
    plot_images()

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Name of the FITS file containing photometry data')
    parser.add_argument('tic_id', type=int, help='TIC ID of the star')
    args = parser.parse_args()

    search_and_extract_info(args.filename, args.tic_id)


if __name__ == "__main__":
    main()
