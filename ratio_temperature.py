#!/usr/bin/env python
import numpy as np
import json
import os
import argparse
from astropy.io import fits


def get_phot_file(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.
    """
    for filename in os.listdir(directory):
        if filename.startswith('phot') and filename.endswith('.fits'):
            return os.path.join(directory, filename)
    raise FileNotFoundError("No photometry file found in the directory.")


def main():
    parser = argparse.ArgumentParser(description='Process photometry FITS files.')
    parser.add_argument('--cam', type=str, default='CMOS', help='Camera type (CMOS or CCD)')
    args = parser.parse_args()

    # Set parameters based on camera type
    if args.cam == 'CMOS':
        APERTURE = 5
        GAIN = 1.13
        EXPOSURE = 10.0
    else:
        APERTURE = 4
        GAIN = 2
        EXPOSURE = 13.0

    # Read the photometry file
    phot_file = get_phot_file('.')
    with fits.open(phot_file) as phot_hdul:
        phot_data = phot_hdul[1].data

        # Extract relevant columns
        tic_ids = phot_data['TIC_ID']
        tmags = phot_data['Tmag'][0]
        teffs = phot_data['Teff'][0]
        flux = phot_data[f'flux_{APERTURE}']

        # Create a dictionary to store aggregated data for each TIC_ID
        tic_data = {}

        for i, tic_id in enumerate(tic_ids):
            if tic_id not in tic_data:
                tic_data[tic_id] = {
                    "flux_values": [],
                    "Tmag": tmags[0],
                    "Teff": teffs[0]
                }
            tic_data[tic_id]["flux_values"].append(flux[i])

        # Filter TIC_IDs with valid Teff and calculate converted flux
        output_data = []
        for tic_id, data in tic_data.items():
            if not np.isnan(data["Teff"]):  # Check for valid Teff
                avg_flux = np.mean(data["flux_values"])  # Calculate average flux
                converted_flux = (avg_flux * GAIN) / EXPOSURE  # Apply conversion
                output_data.append({
                    "TIC_ID": int(tic_id),
                    "Tmag": float(data["Tmag"]),
                    "Teff": float(data["Teff"]),
                    "Converted_Flux": float(converted_flux)
                })

        # Save to JSON file
        output_file = f'flux_vs_temperature_{args.cam}.json'
        with open(output_file, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)

        print(f"Saved {len(output_data)} targets with valid 'Teff' to {output_file}")


if __name__ == "__main__":
    main()