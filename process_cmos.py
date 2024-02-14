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


def get_coords_from_header(fits_file):
    # Open the FITS file
    hdulist = fits.open(fits_file)

    # Get the binary table data
    data = hdulist[1].data

    # Extract RA and DEC coordinates
    RA = data['ra_deg']
    DEC = data['dec_deg']

    # Print the first 10 entries as an example
    for i in range(10):
        print(f'Entry {i + 1}: RA = {RA[i]}, DEC = {DEC[i]}')

    # Close the FITS file
    hdulist.close()


def main():
    # Get the location of the observatory
    site_location, site_topos = get_location()

    # Calibrate the images
    calibrate_images(base_path)

    # Get the coordinates from the header
    fits_file = 'NG0547-0421_catalog.fits'
    get_coords_from_header(fits_file)


if __name__ == "__main__":
    main()
