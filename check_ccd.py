#! /usr/bin/env python

"""
This script checks the headers of the FITS files in the specified directory
and moves the files without CTYPE1 and/or CTYPE2 to a separate directory.

Usage:
python check_headers.py
"""

from datetime import datetime, timedelta
from donuts import Donuts
from astropy.io import fits
import numpy as np
import os
import json
import warnings

warnings.simplefilter('ignore', category=UserWarning)


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break

def filter_filenames(directory):
    """
    Filter filenames based on specific criteria.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        Filtered list of filenames.
    """
    filtered_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            fits_path = os.path.join(directory, filename)
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                if 'IMGCLASS' in header and header['IMGCLASS'] == 'SCIENCE':
                    filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def get_prefix(filenames, directory):
    """
    Extract unique prefixes from a list of filenames based on the OBJECT keyword in FITS headers.

    Parameters
    ----------
    filenames : list of str
        List of filenames.
    directory : str
        Directory containing the FITS files.

    Returns
    -------
    set of str
        Set of unique prefixes extracted from the OBJECT keyword in the FITS headers.
    """
    prefixes = set()
    for filename in filenames:
        fits_path = os.path.join(directory, filename)
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            if 'OBJECT' in header:
                prefix = header['OBJECT']
                prefixes.add(prefix)
    return prefixes


def check_headers(directory, filenames):
    """
    Check headers of all files for CTYPE1 and CTYPE2.

    Parameters
    ----------
    directory : str
        Path to the directory.
    filenames : list of str
        List of filenames.
    """
    no_wcs = os.path.join(directory, 'no_wcs')
    if not os.path.exists(no_wcs):
        os.makedirs(no_wcs)

    for file in filenames:
        try:
            with fits.open(os.path.join(directory, file)) as hdulist:
                header = hdulist[0].header
                ctype1 = header.get('CTYPE1')
                ctype2 = header.get('CTYPE2')

                if ctype1 is None or ctype2 is None:
                    print(f"Warning: {file} does not have CTYPE1 and/or CTYPE2 in the header. Moving to "
                          f"'no_wcs' directory.")
                    new_path = os.path.join(no_wcs, file)
                    os.rename(os.path.join(directory, file), new_path)

        except Exception as e:
            print(f"Error checking header for {file}: {e}")

    print("Done checking headers, number of files without CTYPE1 and/or CTYPE2:", len(os.listdir(no_wcs)))


def check_donuts(directory, filenames):
    """
    Check donuts for each image in the directory.

    Parameters
    ----------
    directory : str
        Directory containing the images.
    filenames : list of str
        List of filenames.
    """
    for filename in filenames:
        # Assuming Donuts class and measure_shift function are defined elsewhere
        fits_path = os.path.join(directory, filename)
        d = Donuts(fits_path)

        shift = d.measure_shift()
        sx = round(shift.x.value, 2)
        sy = round(shift.y.value, 2)
        print(f'{filename} shift X: {sx} Y: {sy}')
        shifts = np.array([abs(sx), abs(sy)])

        if np.sum(shifts > 50) > 0:
            print(f'{filename} image shift too big X: {sx} Y: {sy}')
            if not os.path.exists('failed_donuts'):
                os.mkdir('failed_donuts')
            comm = f'mv {filename} failed_donuts/'
            print(comm)
            os.system(comm)
            

def main():
    # get the current working directory
    parent_directory = os.getcwd()

    # get a list of subdirectories inside the parent directory
    subdirectories = [name for name in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, name))]

    # iterate over each subdirectory
    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_observeField"):
            # form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # set directory for the current subdirectory
            directory = subdirectory_path
            print(f"Directory: {directory}")

            # filter filenames only for .fits data files
            filenames = filter_filenames(directory)
            print(f"Number of files: {len(filenames)}")

            # Check headers for CTYPE1 and CTYPE2
            check_headers(directory, filenames)

            # Check donuts for the current subdirectory
            check_donuts(subdirectory, filenames)

    print("Done.")


if __name__ == "__main__":
    main()
