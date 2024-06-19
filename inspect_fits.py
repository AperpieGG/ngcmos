#!/usr/bin/env python

import argparse
import os
import numpy as np
from astropy.io import fits


def read_fits_file(filename):
    with fits.open(filename) as hdul:
        phot_table = hdul[1].data
    return phot_table


def print_column_info(table, column_name):
    print(f"--- Column: {column_name} ---")
    column_data = table[column_name]
    for i, cell in enumerate(column_data):
        if i >= 5:  # Limit to first 5 entries for brevity
            break
        cell_array = np.array(cell)
        print(f"Row {i}: length = {len(cell_array)}, first 5 values = {cell_array[:5]}")
    print(f"Total rows: {len(column_data)}")
    print()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Inspect FITS table columns')
    parser.add_argument('filename', type=str, help='Name of the FITS file containing photometry data')
    args = parser.parse_args()
    filename = args.filename

    # Get the current night directory
    current_night_directory = os.getcwd()
    file_path = os.path.join(current_night_directory, filename)

    # Read the FITS file
    print(f"Reading the FITS file {filename}...")
    phot_table = read_fits_file(file_path)

    # Print information for ZP, Airmass, and Magnitude columns
    print_column_info(phot_table, 'ZP')
    print_column_info(phot_table, 'Airmass')
    print_column_info(phot_table, 'Magnitude')


if __name__ == "__main__":
    main()
