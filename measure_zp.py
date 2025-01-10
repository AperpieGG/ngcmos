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
    color_list = []

    for tic_id in tic_ids:
        # Average flux for the current TIC ID
        tic_data = table[table['TIC_ID'] == tic_id]

        tic_flux = np.mean(tic_data[f'flux_{APERTURE}'])
        # First Tmag value for the current TIC ID
        tic_Tmag = tic_data['Tmag'][0]
        target_color_index = tic_data['gaiabp'][0] - tic_data['gaiarp'][0]

        # Calculate zero point for the current TIC ID
        zp = tic_Tmag + 2.5 * np.log10(tic_flux / EXPOSURE)
        print(f'TIC ID: {tic_id}, Zero Point: {zp}, Color Index: {target_color_index}')
        zp_list.append(zp)
        color_list.append(target_color_index)

    return zp_list, color_list


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
        zp_list, color_list = measure_zp(phot_table, APERTURE, EXPOSURE)
        print(f"Zero point average: {np.nanmean(zp_list)}")

        # save the results to a json file
        with open(f'zp{APERTURE}.json', 'w') as json_file:
            json.dump(np.nanmean(zp_list), json_file, indent=4)

        print(f"Results saved to zp{APERTURE}.json")

        # Filter out entries where either zp or color is NaN
        valid_data = [{'zp': zp, 'color': color} for zp, color in zip(zp_list, color_list)
                      if not np.isnan(zp) and not np.isnan(color)]

        # Extract zp_list and color_list separately after filtering
        filtered_zp_list = [entry['zp'] for entry in valid_data]
        filtered_color_list = [entry['color'] for entry in valid_data]

        with open(f'zp{APERTURE}_list.json', 'w') as json_file:
            # Save the filtered lists to a JSON file
            json.dump({'zp_list': filtered_zp_list, 'color_list': filtered_color_list}, json_file, indent=4)

        print(f"Results saved to zp{APERTURE}_list.json")


if __name__ == "__main__":
    main()
