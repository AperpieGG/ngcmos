#!/usr/bin/env python3

"""

This script adds headers to the FITS files in the specified directory

Usage:
python adding_headers.py

"""
import argparse
import glob
from datetime import datetime, timedelta
from astropy.io import fits
import os
import numpy as np


def update_header(directory):
    """
    Update the header of FITS files in the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing FITS files.
    """

    for filename in sorted(glob.glob(os.path.join(directory, '*_r.fits'))):
        if 'FILTER' in fits.getheader(filename):
            print(f"Header already present for {filename}")
            continue
        with fits.open(filename, mode='update') as hdul:
            if 'FILTER' not in hdul[0].header:
                hdul[0].header['FILTER'] = 'NGTS'
            else:
                print(f"FILTER already present for {filename}")
            if 'AIRMASS' not in hdul[0].header:
                airmass = 1 / np.cos(np.radians(90 - hdul[0].header['ALTITUDE']))
                hdul[0].header['AIRMASS'] = airmass
            else:
                print(f"AIRMASS already present for {filename}")
            hdul.flush()
            print(f"Updated header for {filename}")
    print("All headers updated")


def main():
    """
    Main function for the script
    """

    parser = argparse.ArgumentParser(description='Add headers to FITS files')
    parser.add_argument('--directory', type=str, help='Path to the directory containing FITS files')
    args = parser.parse_args()

    if args.directory:
        custom_directory = args.directory
        update_header(custom_directory)
        print(f"Using custom directory {custom_directory}")
    else:
        custom_directory = None
        print("No custom directory specified")


if __name__ == "__main__":
    main()

# TODO: add parse arguments to the main function for the directory path
