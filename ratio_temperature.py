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

        # Sort data by airmass (smallest first)
        sorted_idx = np.argsort(data['airmass'])  # Get sorted indices based on airmass
        sorted_data = data[sorted_idx]  # Sort data accordingly

        # Extract 300 unique frame IDs with the lowest airmass values
        unique_frames = []
        frame_airmass = {}

        for i in sorted_idx:  # Iterate through sorted indices
            frame_id = data['frame_id'][i]
            airmass = data['airmass'][i]

            if frame_id not in frame_airmass:
                frame_airmass[frame_id] = airmass
                unique_frames.append(frame_id)

            if len(unique_frames) == 500:  # Stop once we have 300 unique frame IDs
                break

        # Sort frame IDs based on their actual airmass values
        unique_frames.sort(key=lambda x: frame_airmass[x])

        # Print each frame_id and its corresponding airmass value
        print("Selected 500 unique frame_ids with lowest airmass values:")
        for frame in unique_frames:
            print(f"Frame ID: {frame}, Airmass: {frame_airmass[frame]}")

        # Filter data based on selected frame IDs
        data = data[np.isin(data['frame_id'], unique_frames)]
        print(f"Entries for selected frame_ids: {len(data)}")

        # Filter Tmag range (10 < Tmag < 14)
        data = data[(data['Tmag'] < 14) & (data['Tmag'] > 10)]
        print(f"Entries after Tmag filter: {len(data)}")

        # Filter valid Teff
        data = data[~np.isnan(data['Teff'])]
        print(f"Entries after Teff filter: {len(data)}")

        # Extract columns
        flux_col = f'flux_{AP}'
        if flux_col not in data.names:
            raise ValueError(f"Column {flux_col} not found.")

        # Compute flux
        flux = data[flux_col]
        avg_flux = np.mean(flux)

        # Extract other parameters
        tic = data['TIC_ID']
        tmag = data['Tmag']
        teff = data['Teff']
        color = data['gaiabp'] - data['gaiarp']

        # Store data per TIC_ID
        tic_data = {}
        for tid in np.unique(tic):
            mask = tic == tid
            tic_data[tid] = {
                "flux": flux[mask][0],
                "Tmag": tmag[mask][0],
                "Teff": teff[mask][0],
                "Color": color[mask][0]
            }

        # Compute converted flux and save
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

        # Save results
        out_file = f'flux_vs_temp_{args.cam}.json'
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=4)

        print(f"Saved {len(output)} targets to {out_file}")
        print(f"Average flux for selected 300 frame_ids: {avg_flux}")


if __name__ == "__main__":
    main()
