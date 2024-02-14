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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.fromnumeric")
warnings.filterwarnings("ignore", category=UserWarning, module="donuts.image")

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


def check_donuts(filenames):
    """
    Check the donuts for each group of images and measure the shifts.

    """
    excluded_keywords = ['catalog', 'morning', 'evening', 'bias', 'flat', 'dark']
    grouped_filenames = defaultdict(list)
    # Group filenames by their prefixes
    for filename in filenames:
        prefix = get_prefix(filename)
        grouped_filenames[prefix].append(filename)

    # Loop through each group of filenames
    for prefix, filenames in grouped_filenames.items():
        # Use the first image as the reference image
        reference_image = filenames[0]
        d = Donuts(reference_image)

        # Loop through each image in the group
        for filename in filenames[1:]:
            # Check if the filename contains any of the excluded keywords
            if any(keyword in filename for keyword in excluded_keywords):
                # Skip this filename
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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


def get_region(directory, prefix):
    """
    Find the region file corresponding to the given prefix in the current night directory.

    Parameters
    ----------
    directory : str
        Path to the current night directory.
    prefix : str
        Prefix to search for.

    Returns
    -------
    pyregion.ShapeList or None
        Region shapes parsed from the region file if found, otherwise None.
    """
    region_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('_input.reg')]
    print("Found region file {} for prefix {}".format(region_files, prefix))
    if region_files:
        region_file_path = os.path.join(directory, region_files[0])
        try:
            x, y = pyregion.open(region_file_path)
            return x, y
        except Exception as e:
            print(f"Error reading region file {region_file_path}: {e}")
            return None
    else:
        return None


def phot(data, shift, x, y, r_aperture, filename, jd, bjd, hjd,
         phot_filename_prefix="rtp",
         gain=1.12):
    # Apply shifts
    x = np.array(x) - shift.x.value
    y = np.array(y) - shift.y.value

    # Preamble for output
    out_str = "{0:s}  {1:.8f}  {2:.8f}  {3:.8f}  ".format(filename, jd.value, bjd.value, hjd.value)

    # Perform photometry for each star
    for i, j in zip(x, y):
        # Extract flux within the fixed aperture
        flux, fluxerr, _ = sep.sum_circle(data, i, j, r_aperture, subpix=0, gain=gain)

        # Estimate background using SEP
        bkg = sep.Background(data)
        bkg_value = bkg.globalback

        # Adjusted flux to subtract background
        flux_w_sky = flux - bkg_value * np.pi * r_aperture ** 2

        # Compile output string
        temp = "{:.2f}  {:.2f}  {:.2f}  {:.2f}  ".format(
            float(i), float(j), float(flux_w_sky), float(fluxerr))
        out_str += temp

    out_str += "\n"

    # Output the photometry
    with open(f"{phot_filename_prefix}.phot{r_aperture}", 'a') as outfile:
        outfile.write(out_str)


def main():
    # Get the location of the observatory
    site_location, site_topos = get_location()

    # Calibrate the images and get the list of FITS files
    fits_files = calibrate_images(base_path)

    # Get coordinates from the headers of catalog FITS files
    catalog_files = {}
    for prefix in set(get_prefix(filename) for filename in fits_files):
        catalog_files[prefix] = [f for f in fits_files if f.startswith(prefix) and 'catalog' in f]

    # Check donuts for each group
    grouped_filenames = defaultdict(list)
    for filename in fits_files:
        prefix = get_prefix(filename)
        grouped_filenames[prefix].append(filename)

    # Check donuts for each group
    for filenames in grouped_filenames.values():
        check_donuts(filenames)

    # Find region files for each prefix
    current_night_directory = find_current_night_directory(base_path)
    for filenames in grouped_filenames.values():
        for filename in filenames:
            prefix = get_prefix(filename)
            region_file = get_region(current_night_directory, prefix)
            if region_file:
                print(f"Found region file {region_file} for prefix {prefix}")
            else:
                print(f"No region file found for prefix {prefix}")


if __name__ == "__main__":
    main()
