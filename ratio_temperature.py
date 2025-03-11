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


def find_exposure(directory):
    """
    Find exposure time from a FITS file.
    """
    for filename in os.listdir(directory):
        exclude_words = ['phot', 'evening', 'master', 'morning', 'catalog']
        if (filename.endswith('.fits') or filename.endswith('fits.bz2')
                and not any(word in filename for word in exclude_words)):
            with fits.open(filename) as hdul:
                header = hdul[0].header
                if 'EXPTIME' and 'OBJECT' in header:
                    return header['EXPTIME'], header['OBJECT']
    return None  # If no valid exposure time found


def main():
    parser = argparse.ArgumentParser(description='Process photometry FITS files.')
    parser.add_argument('--cam', type=str, default='CMOS', help='Camera type (CMOS or CCD)')
    args = parser.parse_args()

    # Set parameters based on camera type
    AP = 5 if args.cam == 'CMOS' else 4
    GAIN = 1.13 if args.cam == 'CMOS' else 2

    # Read the photometry file
    print("Locating photometry file...")
    phot_file = get_phot_file('.')
    print(f"Photometry file found: {phot_file}")

    # Get exposure time
    exp_result = find_exposure('.')
    if exp_result is None or len(exp_result) != 2:
        raise ValueError("find_exposure('.') did not return (EXPOSURE, OBJECT).")
    EXP, OBJ = exp_result
    print(f"Exposure: {EXP} sec for {OBJ}")

    with fits.open(phot_file) as hdul:
        data = hdul[1].data

        # **Step 1: Identify Frame IDs in Airmass Peak**
        min_airmass, max_airmass = 1.2, 1.2  # Define the peak airmass range
        mask = (data['airmass'] >= min_airmass) & (data['airmass'] <= max_airmass)

        if np.sum(mask) == 0:
            raise ValueError("No frames found in the specified airmass range (1.2 to 1.2).")

        frame_ids = np.unique(data['frame_id'][mask])
        print(f"Found {len(frame_ids)} frame IDs in airmass range 1.2 to 1.2.")

        # **Step 2: Filter Data for Selected Frame IDs**
        filtered_data = data[np.isin(data['frame_id'], frame_ids)]
        print(f"Entries after filtering frame IDs: {len(filtered_data)}")

        # **Step 3: Apply Star Selection Criteria**
        filtered_data = filtered_data[(filtered_data['Tmag'] < 14) & (filtered_data['Tmag'] > 10)]
        print(f"Entries after Tmag filter (10 < Tmag < 14): {len(filtered_data)}")

        filtered_data = filtered_data[~np.isnan(filtered_data['Teff'])]
        print(f"Entries after Teff filter: {len(filtered_data)}")

        # **Step 4: Extract Flux and Compute Statistics**
        flux_col = f'flux_{AP}'
        if flux_col not in filtered_data.names:
            raise ValueError(f"Column {flux_col} not found.")

        flux = filtered_data[flux_col]
        avg_flux = np.mean(flux)
        print(f"Average flux: {avg_flux}")

        # **Step 5: Extract Other Information**
        tic_ids = filtered_data['TIC_ID']
        tmags = filtered_data['Tmag']
        teffs = filtered_data['Teff']
        colors = filtered_data['gaiabp'] - filtered_data['gaiarp']

        # **Step 6: Organize Data by TIC_ID**
        tic_data = {}
        for tid in np.unique(tic_ids):
            mask = tic_ids == tid
            tic_data[tid] = {
                "flux": flux[mask][0],
                "Tmag": tmags[mask][0],
                "Teff": teffs[mask][0],
                "Color": colors[mask][0]
            }

        # **Step 7: Compute Converted Flux and Save**
        output = []
        for tid, vals in tic_data.items():
            conv_flux = (vals["flux"] * GAIN) / EXP
            output.append({
                "TIC_ID": int(tid),
                "Tmag": float(vals["Tmag"]),
                "Teff": float(vals["Teff"]),
                "Color": float(vals["Color"]),
                "Converted_Flux": float(conv_flux)
            })

        # **Step 8: Save Data**
        out_file = f'flux_vs_temperature_{args.cam}.json'
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=4)

        print(f"Saved {len(output)} targets to {out_file}")
        print(f"Average flux for selected frames: {avg_flux}")


if __name__ == "__main__":
    main()