#! /usr/bin/env python
import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from astropy.coordinates import EarthLocation
from skyfield.api import Topos
import re
from calibration_images import reduce_images, bias, dark, flat
from donuts import Donuts
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.fromnumeric")
warnings.filterwarnings("ignore", category=UserWarning, module="donuts.image")


# pylint: disable = invalid-name
# pylint: disable = redefined-outer-name
# pylint: disable = no-member
# pylint: disable = too-many-locals
# pylint: disable = too-many-arguments
# pylint: disable = unused-variable

# Define directories
calibration_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/20231212/'
base_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/'
out_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/calibration_images/'

calibration_path_2 = '/home/ops/data/20231212/'
base_path_2 = '/home/ops/data/'
out_path_2 = '/home/ops/data/calibration_images/'

# Select directory based on existence
if os.path.exists(base_path_1):
    calibration_path = calibration_path_1
    base_path = base_path_1
    out_path = out_path_1
else:
    base_path = base_path_2
    calibration_path = calibration_path_2
    out_path = out_path_2


# Find current night directory
def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    if not then use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str or None
        Path to the current night directory if found, otherwise None.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return os.getcwd()


# Extract prefix from filename
def get_prefix(filename):
    """
    Extract prefix from filename
    """
    return filename[:11]


# Calibrate images
def calibrate_images(directory):
    """
     Reduce the images in the specified directory.

     Parameters
     ----------
     directory : str
         Base path for the directory.
     master_bias : numpy.ndarray
         Master bias.
     master_dark : numpy.ndarray
         Master dark.
     master_flat : numpy.ndarray
         Master flat.

     Returns
     -------
     None
     """
    current_night_directory = find_current_night_directory(directory)
    master_bias = bias(calibration_path, out_path)
    master_dark = dark(calibration_path, out_path, master_bias)
    master_flat = flat(base_path, out_path, master_bias, master_dark)
    fits_files = [f for f in os.listdir(current_night_directory) if f.endswith('.fits') and 'catalog' not in f]
    reduce_images(current_night_directory, master_bias, master_dark, master_flat)
    return fits_files


# Check donuts for each group
def check_donuts(filenames):
    """
    Check donuts for each group.

    """
    excluded_keywords = ['catalog', 'morning', 'evening', 'bias', 'flat', 'dark']
    grouped_filenames = defaultdict(list)
    for filename in filenames:
        prefix = get_prefix(filename)
        grouped_filenames[prefix].append(filename)

    for prefix, filenames in grouped_filenames.items():
        filenames.sort()

        reference_image = filenames[0]
        d = Donuts(reference_image)
        for filename in filenames[1:]:
            if any(keyword in filename for keyword in excluded_keywords):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
                continue


# Find region files for each prefix
def get_region_files(filenames):
    """
    Find region files for each prefix.

    Parameters
    ----------
    filenames : list of filenames to search for region files.

    Returns
    -------
    dict
        Dictionary containing prefixes as keys and region files as values.
    """
    grouped_region_files = defaultdict(set)  # Use set to ensure uniqueness

    for filename in filenames:
        prefix = get_prefix(filename)
        exclude_keywords = ['catalog', 'morning', 'evening', 'bias', 'flat', 'dark']
        if any(keyword in filename for keyword in exclude_keywords):
            continue
        region_files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith('_input.reg')]
        grouped_region_files[prefix].update(region_files)

    return grouped_region_files


def read_region_files(region_files):
    """
    Read the contents of region files.

    Parameters
    ----------
    region_files : list
        List of region files to read.

    Returns
    -------
    dict
        Dictionary containing region file names as keys and their contents as values.
    """
    region_contents = {}

    for region_file in region_files:
        with open(region_file, 'r') as file:
            contents = file.read()
            region_contents[region_file] = contents

    return region_contents


# Main function
def main():
    # Get observatory location
    site_location, site_topos = get_location()

    # Calibrate images and get FITS files
    fits_files = calibrate_images(base_path)

    # Check donuts for each group
    check_donuts(fits_files)

    # Get region files for each prefix
    region_files = get_region_files(fits_files)

    # Print region files for each prefix
    for prefix, files in region_files.items():
        print(f"Prefix: {prefix}, Region Files: {files}")

    # Read the contents of region files
    region_contents = {}
    for prefix, files in region_files.items():
        region_contents[prefix] = read_region_files(files)

    # Print the contents of region files
    for prefix, contents in region_contents.items():
        print(f"Prefix: {prefix}")


if __name__ == "__main__":
    main()
