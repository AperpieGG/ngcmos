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

        # Find the `frame_id` with the smallest `airmass`
        unique_frames = np.unique(phot_data['frame_id'])
        frame_airmass = {
            frame: np.min(phot_data[phot_data['frame_id'] == frame]['airmass']) for frame in unique_frames
        }
        best_frame_id = min(frame_airmass, key=frame_airmass.get)
        print(f"Selected frame_id with smallest airmass: {best_frame_id}")

        # Filter data to include only the selected frame_id
        phot_data = phot_data[phot_data['frame_id'] == best_frame_id]
        print(f"Number of entries for selected frame_id: {len(phot_data)}")

        # Filter stars with Tmag < 14 and Tmag > 10
        phot_data = phot_data[(phot_data['Tmag'] < 14) & (phot_data['Tmag'] > 10)]
        print(f"Number of entries after filtering Tmag (10 < Tmag < 14): {len(phot_data)}")

        # Filter stars with valid Teff (not NaN or null)
        phot_data = phot_data[~np.isnan(phot_data['Teff'])]
        print(f"Number of entries after filtering valid Teff: {len(phot_data)}")

        # Extract relevant columns
        tic_ids = phot_data['TIC_ID']
        flux_w_sky_col = f'flux_w_sky_{APERTURE}'  # add if solve for sky
        flux_col = f'flux_{APERTURE}'

        if flux_w_sky_col not in phot_data.names or flux_col not in phot_data.names:
            raise ValueError(f"Columns {flux_w_sky_col} or {flux_col} not found in the photometry file.")

        # Calculate sky background
        sky_background = phot_data[flux_w_sky_col] - phot_data[flux_col]
        # fluxes = sky_background  # Use sky background

        fluxes = phot_data[flux_col]  # Use flux
        tmags = phot_data['Tmag']
        teffs = phot_data['Teff']
        COLORs = phot_data['gaiabp'] - phot_data['gaiarp']

        # Print unique TIC_IDs for the analysis
        unique_tic_ids = np.unique(tic_ids)
        print(f"Unique TIC_IDs for the analysis: {len(unique_tic_ids)}")
        print(f"TIC_IDs: {unique_tic_ids}")

        # Create a dictionary to store data grouped by TIC_ID
        tic_data = {}
        for tic_id in unique_tic_ids:
            mask = phot_data['TIC_ID'] == tic_id
            target_flux = fluxes[mask][0]  # Single flux value for the selected frame
            tmag = tmags[mask][0]
            teff = teffs[mask][0]
            COLOR = COLORs[mask][0]

            tic_data[tic_id] = {
                "flux_value": target_flux,
                "Tmag": tmag,
                "Teff": teff,
                "COLOR": COLOR
            }

        # Filter TIC_IDs with valid Teff and calculate converted flux
        output_data = []
        for tic_id, data in tic_data.items():
            converted_flux = (data["flux_value"] * GAIN) / EXPOSURE  # e-/s
            output_data.append({
                "TIC_ID": int(tic_id),
                "Tmag": float(data["Tmag"]),
                "Teff": float(data["Teff"]),
                "COLOR": float(data["COLOR"]),
                "Converted_Flux": float(converted_flux)
            })

        # Save to JSON file
        output_file = f'flux_vs_temperature_{args.cam}.json'
        with open(output_file, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)

        print(f"Saved {len(output_data)} targets with valid 'Teff' to {output_file}")


if __name__ == "__main__":
    main()
