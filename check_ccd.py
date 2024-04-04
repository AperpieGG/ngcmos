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


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    If not found, use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str
        Path to the current night directory.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    return current_date_directory if os.path.isdir(current_date_directory) else os.getcwd()


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
    Extract unique prefixes from a list of filenames.

    Parameters
    ----------
    filenames : list of str
        List of filenames.
    directory : str
        Directory containing the FITS files.

    Returns
    -------
    set of str
        Set of unique prefixes extracted from the filenames.
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


def check_donuts(file_groups, filenames):
    """
    Check donuts for each group of images with the same prefix.

    Parameters
    ----------
    file_groups : list of str
        Prefixes for the groups of images.
    filenames : list of str
        List of lists of filenames for the groups of images.
    """
    for filename, file_group in zip(filenames, file_groups):
        # Using the first filename as the reference image
        reference_image = file_group[0]
        print(f"Reference image: {reference_image}")

        # Assuming Donuts class and measure_shift function are defined elsewhere
        d = Donuts(reference_image)

        for filename in file_group[1:]:
            shift = d.measure_shift(filename)
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
    # set directory for the current night or use the current working directory
    directory = find_current_night_directory(base_path)
    print(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    print(f"Number of files: {len(filenames)}")

    # Iterate over each filename to get the prefix
    prefixes = get_prefix(filenames, directory)
    print(f"The prefixes are: {prefixes}")

    # Get filenames corresponding to each prefix
    prefix_filenames = [[filename for filename in filenames if filename.startswith(prefix)] for prefix in prefixes]

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames)

    # Check donuts for each group
    check_donuts(prefix_filenames, filenames)

    print("Done.")


if __name__ == "__main__":
    main()
