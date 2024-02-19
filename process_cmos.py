#! /usr/bin/env python
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from astropy.wcs import WCS
from skyfield.api import Topos
import re
from calibration_images import reduce_images
from donuts import Donuts
from utils import source_extract, catalogue_to_pixels
import warnings
from astropy.io import fits

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.fromnumeric")
warnings.filterwarnings("ignore", category=UserWarning, module="donuts.image")
warnings.filterwarnings("ignore", category=UserWarning, module="FITsFixedWarning")

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


def find_first_image_of_each_prefix(filenames):
    # Sort the filenames
    sorted_filenames = sorted(filenames)

    # Create a dictionary to store the first image of each prefix
    first_images = {}

    # Iterate over each filename
    for filename in sorted_filenames:
        # Extract the prefix from the filename
        prefix = get_prefix(filename)

        # Check if the prefix is not already in the dictionary
        if prefix not in first_images:
            # Add the filename to the dictionary
            first_images[prefix] = filename

    print(first_images)
    return first_images


def get_prefix(filename):
    """
    Extract prefix from filename
    """
    return filename[:11]


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


def parse_region_content(region_content):
    """
    Parse the content of a region file to extract RA and Dec coordinates.

    Parameters
    ----------
    region_content : str
        Content of the region file.

    Returns
    -------
    list of tuples
        List of (RA, Dec) coordinate tuples extracted from the region content.
    """
    ra_dec_coords = []
    # Split the region content into lines
    lines = region_content.split('\n')
    for line in lines:
        # Assuming each line represents a circular region with RA and Dec coordinates
        # Extract the RA and Dec coordinates from the line
        if line.startswith('circle'):
            # Example line format: "circle(123.456, 78.901, 2.0)"
            parts = line.split('(')[1].split(')')[0].split(',')
            ra = float(parts[0])
            dec = float(parts[1])
            ra_dec_coords.append((ra, dec))
    return ra_dec_coords


def main():
    # Calibrate images and get FITS files
    reduced_data, jd_list, bjd_list, hjd_list, filenames = reduce_images(base_path, out_path)

    # Check donuts for each group
    check_donuts(filenames)

    # Get region files for each prefix
    region_files = get_region_files(filenames)

    # Print region files for each prefix
    for prefix, files in region_files.items():
        print(f"Prefix: {prefix}, Region Files: {files}")

    # Read the contents of region files
    region_contents = {}
    for prefix, files in region_files.items():
        region_contents[prefix] = read_region_files(files)

    for prefix, contents in region_contents.items():
        first_images = find_first_image_of_each_prefix(filenames)
        for region_file, region_content in contents.items():
            # Extract RA and Dec from region content (assuming you have a function to parse the region file)
            ra_dec_coords = parse_region_content(region_content)
            print(f"Prefix: {prefix}, Region File: {region_file}, cordinates: {ra_dec_coords}")

            if prefix in first_images:  # Check if the prefix exists in the fits_files dictionary
                # Convert RA and Dec to pixel coordinates
                xy_coordinates = catalogue_to_pixels(first_images[prefix], ra_dec_coords)
                print("X coordinates:", xy_coordinates[0])
                print("Y coordinates:", xy_coordinates[1])
            else:
                print(f"No image found for prefix {prefix}")


if __name__ == "__main__":
    main()

# TODO: Add aditional annulus size for the region files (follow J MCMC code)
