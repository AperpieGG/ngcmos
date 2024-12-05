#! /usr/bin/env python
from astropy.io import fits
import numpy as np
import os
import argparse


def get_phot_file(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the file.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    return [os.path.join(directory, filename) for filename in os.listdir(directory)
            if filename.startswith('phot') and filename.endswith('.fits')]


def multiply_flux_values(file_path, gain):
    """
    Multiply all flux-related values in the FITS file by the provided gain.

    Parameters
    ----------
    file_path : str
        Path to the FITS file.
    gain : float
        Gain to multiply the flux values.
    """
    try:
        with fits.open(file_path, mode='update') as hdul:
            phot_table = hdul[1].data  # Assuming the table is in the first extension

            # Find all flux-related fields dynamically
            flux_fields = [name for name in phot_table.names if name.startswith('flux_')
                           or name.startswith('flux_w_sky_') or name.startswith('max_pixel_value_')]

            # Multiply each flux field by the gain
            for field in flux_fields:
                original_values = phot_table[field]
                updated_values = original_values * gain
                phot_table[field][:] = updated_values
                print(f"Updated {field} in {file_path}: multiplied by {gain}")

            # Save changes to the FITS file
            hdul.flush()
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Multiply flux values in photometry FITS files.')
    parser.add_argument('gain', type=float, help='Gain to convert fluxes from ADU to electrons')
    args = parser.parse_args()

    # Get all photometry FITS files in the current directory
    fits_files = get_phot_file('.')

    if not fits_files:
        print("No photometry FITS files found in the current directory.")
        return

    # Process each file
    for file_path in fits_files:
        print(f"Processing file: {file_path}")
        multiply_flux_values(file_path, args.gain)


if __name__ == "__main__":
    main()




