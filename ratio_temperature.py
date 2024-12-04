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
    parser.add_argument('--cam', type=float, help='Gain to convert fluxes from ADU to electrons')
    args = parser.parse_args()
    if args.cam == 'CMOS':
        APERTURE = 5
        GAIN = 1.13
        EXPOSURE = 10.0
    else:
        APERTURE = 4
        GAIN = 2
        EXPOSURE = 13.0
    phot_file = get_phot_file('.')

    # Read the photometry file
    with fits.open(phot_file) as phot_hdul:
        phot_data = phot_hdul[1].data

        # Extract relevant columns
        tic_ids = phot_data['TIC_ID']
        tmags = phot_data['Tmag']
        teffs = phot_data['Teff']
        flux = phot_data[f'flux_{APERTURE}']

        # Calculate the converted flux: (flux_5 * 2) / exposure
        converted_flux = (np.mean(flux) * GAIN) / EXPOSURE

        # Filter rows where 'Teff' is not NULL or nan
        valid_indices = ~np.isnan(teffs)
        valid_tic_ids = tic_ids[valid_indices]
        valid_tmags = tmags[valid_indices]
        valid_teffs = teffs[valid_indices]
        valid_flux = converted_flux[valid_indices]

        # Prepare data for JSON output
        output_data = []
        for i in range(len(valid_tic_ids)):
            output_data.append({
                "TIC_ID": int(valid_tic_ids[i]),
                "Tmag": float(valid_tmags[i]),
                "Teff": float(valid_teffs[i]),
                "Converted_Flux": float(valid_flux[i])
            })

        # Save to JSON file
        output_file = f'flux_vs_temperature_{args.cam}.json'
        with open(output_file, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)

        print(f"Saved {len(output_data)} targets with valid 'Teff' to {output_file}")


if __name__ == "__main__":
    main()
