#!/usr/bin/env python
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_phot_files, read_phot_file, plot_images

plot_images()


def measure_zp(table, APERTURE, EXPOSURE):
    tic_ids = np.unique(table['TIC_ID'])
    print(f'Found {len(tic_ids)} unique TIC IDs')
    zp_list = []

    for tic_id in tic_ids:
        # Average flux for the current TIC ID
        tic_flux = np.mean(table[table['TIC_ID'] == tic_id][f'flux_{APERTURE}'])
        # First Tmag value for the current TIC ID
        tic_Tmag = table[table['TIC_ID'] == tic_id]['Tmag'][0]

        # Calculate zero point for the current TIC ID
        zp = tic_Tmag + 2.5 * np.log10(tic_flux / EXPOSURE)
        print(f'TIC ID: {tic_id}, Zero Point: {zp}')
        zp_list.append(zp)

    return zp_list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Read and organize TIC IDs with associated '
                    'RMS, Sky, Airmass, ZP, and Magnitude from FITS table.'
                    'Example usage if you have CMOS: RN=1.56, DC=1.6, Aper=4, Exp=10.0, Bin=1'
                    'Example usage if you have CCD: RN=12.6, DC=0.00515, Aper=4, Exp=10.0, Bin=1')
    parser.add_argument('--exp', type=float, default=10.0, help='Exposure time in seconds')
    parser.add_argument('--aper', type=str, default=6, help='Aperture size in meters')
    args = parser.parse_args()
    EXPOSURE = args.exp
    APERTURE = args.aper  # Aperture size for the telescope

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
        zp_list = measure_zp(phot_table, APERTURE, EXPOSURE)
        print(f"Zero point average: {np.nanmean(zp_list)}")

        # plot zp_list on a histogram
        plt.hist(zp_list, bins=100, label=f"Avg: {np.nanmean(zp_list):.2f}")
        plt.axvline(np.nanmean(zp_list), color='black', linestyle='--', label=f"Mean: {np.nanmean(zp_list):.2f}")
        # plt.axvline(np.nanmedian(zp_list), color='g', linestyle='--', label=f"Median: {np.nanmedian(zp_list):.2f}")
        plt.xlabel('Zero Point')
        plt.ylabel('Frequency')
        plt.yscale('log')
        # plt.legend(loc='upper right')
        plt.ylim(18, 21)
        plt.savefig(f'zp{APERTURE}.pdf', dpi=300)
        plt.show()

        # save the results to a json file
        with open(f'zp{APERTURE}.json', 'w') as json_file:
            json.dump(np.nanmean(zp_list), json_file, indent=4)


if __name__ == "__main__":
    main()
