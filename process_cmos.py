#!/usr/bin/env python

import os
import sys
import math
from datetime import datetime, timedelta
import sep
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import EarthLocation
from skyfield.api import Topos
import re
import pyregion
from collections import defaultdict
from calibration_images import reduce_images, bias, dark, flat


# pylint: disable = invalid-name
# pylint: disable = redefined-outer-name
# pylint: disable = no-member
# pylint: disable = too-many-locals
# pylint: disable = too-many-arguments
# pylint: disable = unused-variable
# pylint: disable = line-too-long
# pylint: disable = logging-fstring-interpolation

# First directory
calibration_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/20231212/'
base_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/'
out_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/calibration_images/'

# Second directory
calibration_path_2 = '/home/ops/data/20231212/'
base_path_2 = '/home/ops/data/'
out_path_2 = '/home/ops/data/calibration_images/'

# Check if the first directory exists
if os.path.exists(base_path_1):
    calibration_path = calibration_path_1
    base_path = base_path_1
    out_path = out_path_1
else:
    base_path = base_path_2
    calibration_path = calibration_path_2
    out_path = out_path_2


def get_location():
    """
    Get the location of the observatory

    Parameters
    ----------
    None

    Returns
    -------
    site_location : EarthLocation
        location of the observatory

    Raises
    ------
    None
    """
    site_location = EarthLocation(
        lat=-24.615662 * u.deg,
        lon=-70.391809 * u.deg,
        height=2433 * u.m)

    site_topos = Topos(
        latitude_degrees=site_location.lat.to(u.deg).value,
        longitude_degrees=site_location.lon.to(u.deg).value,
        elevation_m=site_location.height.to(u.m).value)

    return site_location, site_topos


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

    # Get the previous date directory in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(directory, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        # Use the current working directory
        return os.getcwd()


def calibrate_images(directory):
    """
    Calibrate the images in the directory

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    None
    """

    # Find the current night directory
    current_night_directory = find_current_night_directory(directory)

    master_bias = bias(calibration_path, out_path)
    master_dark = dark(calibration_path, out_path, master_bias)
    master_flat = flat(base_path, out_path, master_bias, master_dark)

    # Reduce the images
    reduce_images(current_night_directory, master_bias, master_dark, master_flat)


def convert_region_file(input_file, output_file, catalog_file):
    """
    Convert a region file from one format to another and label annuli with TIC IDs.

    Parameters
    ----------
    input_file : str
        Path to the input region file.
    output_file : str
        Path to the output region file.
    catalog_file : str
        Path to the catalog.fits file containing TIC IDs and coordinates.

    Returns
    -------
    None
    """
    # Read TIC IDs and coordinates from catalog file
    with fits.open(catalog_file) as hdul:
        ra = hdul[1].data['pmRA']
        dec = hdul[1].data['pmDEC']
        tic_id = hdul[1].data['ID']

    # Read lines from input region file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for i, line in enumerate(lines):
        if line.startswith('point'):
            # Extract RA and Dec from point line
            parts = line.split(',')
            ra_dec = parts[0].split('(')[1].split(')')
            ra_point, dec_point = float(ra_dec[0].split()[0]), float(ra_dec[1])

            # Find the closest TIC ID based on RA and Dec
            distances = np.sqrt((ra - ra_point)**2 + (dec - dec_point)**2)
            closest_index = np.argmin(distances)
            closest_tic_id = tic_id[closest_index]

            # Create annulus line with TIC ID label
            converted_line = f"annulus({ra_point}, {dec_point}, 20.0, 30.0)  # text={{{closest_tic_id}}}\n"
            converted_lines.append(converted_line)

    # Write converted lines to output file
    with open(output_file, 'w') as f:
        f.writelines(converted_lines)


# Example usage:
input_file = 'NG0547-0421_catalog_master.reg'
output_file = 'testing_NG0547-0421_catalog_master.reg'
catalog_file = 'NG0547-0421_catalog_filtered.fits'
convert_region_file(input_file, output_file, catalog_file)

