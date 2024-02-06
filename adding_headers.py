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


def find_current_night_directory(file_path):
    """
    Find the directory for the current night based on the current date.

    Parameters
    ----------
    file_path : str
        Base path for the directory.

    Returns
    -------
    str or None
        Path to the current night directory if found, otherwise None.
    """

    # Get the current date in the format YYYYMMDD
    current_date = datetime.now().strftime("%Y%m%d") + '/'
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d") + '/'

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(file_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return None


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

# TODO: Add a function to check if the header is already present and skip the file if it is
# TODO: add parse arguments to the main function for the directory path
