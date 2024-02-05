#!/usr/bin/env python3

"""

This script adds headers to the FITS files in the specified directory

Usage:
python adding_headers.py

"""
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


def update_header(data_path):
    """
    Update the header of FITS files in the specified directory.

    Parameters
    ----------
    data_path : str
        Path to the directory containing FITS files.
    """

    files = glob.glob(data_path + 'NG*.fits')
    for f in files:
        with fits.open(f, mode='update') as hdul:
            if 'FILTER' not in hdul[0].header:
                hdul[0].header['FILTER'] = 'NGTS'


def main():
    data_path = '/home/ops/data/testing_photo/'
    update_header(data_path)


if __name__ == "__main__":
    main()
