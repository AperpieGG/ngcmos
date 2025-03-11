#!/usr/bin/env python
import numpy as np
import json
import os
import argparse
from astropy.io import fits


# Script is currently running for sky and not for flux, adjust it accordingly.

def get_phot_file(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.
    """
    for filename in os.listdir(directory):
        if filename.startswith('phot') and filename.endswith('.fits'):
            return os.path.join(directory, filename)
    raise FileNotFoundError("No photometry file found in the directory.")


def find_exposure(directory):
    # search for any .fits file that has a header and contains the EXPTIME keyword
    for filename in os.listdir(directory):
        exclude_words = ['phot', 'evening', 'master', 'morning', 'catalog']
        if (filename.endswith('.fits') or filename.endswith('fits.bz2')
                and not any(word in filename for word in exclude_words)):
            with fits.open(filename) as hdul:
                header = hdul[0].header
                if 'EXPTIME' and 'OBJECT' in header:
                    return header['EXPTIME'], header['OBJECT']


def main():
    parser = argparse.ArgumentParser(description='Process photometry FITS files.')
    parser.add_argument('--cam', type=str, default='CMOS', help='Camera type (CMOS or CCD)')
    args = parser.parse_args()

    # Set parameters based on camera type
    if args.cam == 'CMOS':
        APERTURE = 5
        GAIN = 1.13
    else:
        APERTURE = 4
        GAIN = 2

    # Read the photometry file
    print("Locating photometry file...")
    phot_file = get_phot_file('.')
    print(f"Photometry file found: {phot_file}")

    # Find the exposure time
    EXPOSURE, OBJECT = find_exposure('.')
    print(f"Exposure time found: {EXPOSURE} seconds for {OBJECT}")

    with fits.open(phot_file) as phot_hdul:
        phot_data = phot_hdul[1].data

        # Sort by airmass and extract 41 values around the minimum
        sorted_indices = np.argsort(phot_data['airmass'])
        min_index = sorted_indices[0]  # Get the index of the smallest airmass

        # Select ±20 data points around the min airmass
        lower_bound = max(0, min_index - 20)
        upper_bound = min(len(phot_data), min_index + 21)  # Include 20 above + 1 min + 20 below

        selected_data = phot_data[sorted_indices[lower_bound:upper_bound]]
        print(f"Selected {len(selected_data)} data points around the minimum airmass.")

        # Extract necessary columns
        flux_col = f'flux_{APERTURE}'
        if flux_col not in phot_data.names:
            raise ValueError(f"Column {flux_col} not found in the photometry file.")

        fluxes = selected_data[flux_col]
        avg_flux = np.mean(fluxes)  # Compute the average flux

        # Prepare output data
        output_data = {
            "Min_Airmass": selected_data['airmass'][20],  # The actual minimum airmass value
            "Avg_Flux": avg_flux,
            "Selected_TIC_IDs": selected_data['TIC_ID'].tolist(),
            "Selected_Tmags": selected_data['Tmag'].tolist(),
            "Selected_Teffs": selected_data['Teff'].tolist(),
            "Selected_Colors": (selected_data['gaiabp'] - selected_data['gaiarp']).tolist()
        }

        # Save to JSON file
        output_file = f'flux_vs_temperature_min_airmass_{args.cam}.json'
        with open(output_file, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)

        print(f"Saved data for minimum airmass region to {output_file}")

if __name__ == "__main__":
    main()
