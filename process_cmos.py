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
from donuts import Donuts

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
    list
        List of FITS files generated after calibration.
    """

    # Find the current night directory
    current_night_directory = find_current_night_directory(directory)

    master_bias = bias(calibration_path, out_path)
    master_dark = dark(calibration_path, out_path, master_bias)
    master_flat = flat(base_path, out_path, master_bias, master_dark)

    # Get all FITS files in the directory
    fits_files = [f for f in os.listdir(current_night_directory) if f.endswith('.fits') and 'catalog' not in f]

    # Reduce the images
    reduce_images(current_night_directory, master_bias, master_dark, master_flat)

    # Return the list of filenames
    return fits_files


def get_coords_from_header(fits_files):
    """
    Get the coordinates from the header of FITS files.

    Parameters
    ----------
    fits_files : list
        List of FITS files.

    Returns
    -------
    None
    """
    for fits_file in fits_files:
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


def get_prefix(filename):
    """
    Extracts the prefix (first 11 letters) from the filename.

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
    str
        Prefix of the filename.
    """
    return filename[:11]


# load aperture file

# check donuts and measure shifts
def check_donuts(filenames):
    first_image = filenames[0]
    d = Donuts(first_image)
    for filename in filenames[1:]:

        shift = d.measure_shift(filename)
        sx = round(shift.x.value, 2)
        sy = round(shift.y.value, 2)
        print(f'{filename} shift X: {sx} Y: {sy}')
        # check for big shifts
        shifts = np.array([abs(sx), abs(sy)])
        if np.sum(shifts > 50) > 0:
            print(f'{filename} image shift too big X: {sx} Y: {sy}')
            if not os.path.exists('failed_donuts'):
                os.mkdir('failed_donuts')
            comm = f'mv {filename} failed_donuts/'
            print(comm)
            os.system(comm)
            continue


# photometry

def main():
    # Get the location of the observatory
    site_location, site_topos = get_location()

    # Calibrate the images and get the list of FITS files
    fits_files = calibrate_images(base_path)

    # Get coordinates from the headers of catalog FITS files
    # Get coordinates from the headers of catalog FITS files
    catalog_files = {}
    for prefix in set(get_prefix(filename) for filename in fits_files):
        catalog_files[prefix] = [f for f in fits_files if f.startswith(prefix) and 'catalog' in f]

    # Process catalog files for each prefix
    for prefix, files in catalog_files.items():
        get_coords_from_header(files)

    # Check donuts for each group
    grouped_filenames = defaultdict(list)
    for filename in fits_files:
        prefix = get_prefix(filename)
        grouped_filenames[prefix].append(filename)

    # Check donuts for each group
    for filenames in grouped_filenames.values():
        check_donuts(filenames)


if __name__ == "__main__":
    main()
