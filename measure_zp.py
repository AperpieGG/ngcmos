#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_phot_files, read_phot_file, plot_images

plot_images()


def measure_zp(table, APERTURE, GAIN, EXPOSURE):
    tic_ids = np.unique(table['TIC_ID'])
    print(f'Found {len(tic_ids)} unique TIC IDs')
    zp_list = []
    for tic_id in tic_ids:
        # average fluxes for each TIC ID
        tic_fluxes = np.mean(table[table['TIC_ID'] == tic_id][f'flux_{APERTURE}'])
        # first Tmag for each TIC ID, it's the same for all fluxes
        tic_Tmags = table[table['TIC_ID'] == tic_id]['Tmag'][0]

    for i, tic_flux in enumerate(tic_fluxes):
        # calculate zero point for each TIC ID
        zp = tic_Tmags[i] - 2.5 * np.log10(tic_flux / EXPOSURE)
        print(f'TIC ID: {tic_id}, Zero Point: {zp}')
        zp_list.append(zp)

    # average the zero points
    zp_avg = np.mean(zp_list)

    return zp_avg, zp_list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Read and organize TIC IDs with associated '
                    'RMS, Sky, Airmass, ZP, and Magnitude from FITS table.'
                    'Example usage if you have CMOS: RN=1.56, DC=1.6, Aper=4, Exp=10.0, Bin=1'
                    'Example usage if you have CCD: RN=12.6, DC=0.00515, Aper=4, Exp=10.0, Bin=1')
    parser.add_argument('--exp', type=float, default=10.0, help='Exposure time in seconds')
    parser.add_argument('--aper', type=float, default=6, help='Aperture size in meters')
    parser.add_argument('--gain', type=float, default=1.13, help='Gain in electrons per ADU')
    args = parser.parse_args()
    EXPOSURE = args.exp
    APERTURE = args.aper  # Aperture size for the telescope
    GAIN = args.gain  # Gain in electrons per ADU

    # Get the current night directory
    current_night_directory = os.getcwd()

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        print(f"Photometry file: {phot_file}")

        # Measure zero point
        zp_avg, zp_list = measure_zp(phot_table, APERTURE, GAIN, EXPOSURE)

        print(f"Zero point average: {zp_avg}")

        # plot zp_list on a histogram
        plt.hist(zp_list, bins=100)
        plt.xlabel('Zero Point')
        plt.ylabel('Frequency')
        plt.title('Zero Point Histogram')
        plt.show()


if __name__ == "__main__":
    main()
