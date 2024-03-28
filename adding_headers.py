#!/usr/bin/env python3

"""

This script adds headers to the FITS files in the specified directory

Usage:
python adding_headers.py

"""
import argparse
from astropy.io import fits
import os
import numpy as np
from astropy.time import Time
import astropy.units as u
from utils import get_location, get_light_travel_times


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
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def update_header(directory):
    """
    Update the header of FITS files in the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing FITS files.
    """

    for filename in filter_filenames(directory):
        filename = os.path.join(directory, filename)
        with fits.open(filename, mode='update') as hdul:
            if 'FILTER' not in hdul[0].header:
                hdul[0].header['FILTER'] = 'NGTS'
            else:
                print(f"FILTER already present for {filename}")
            if 'AIRMASS' not in hdul[0].header:
                airmass = 1 / np.cos(np.radians(90 - hdul[0].header['ALTITUDE']))
                hdul[0].header['AIRMASS'] = airmass
            else:
                print(f"AIRMASS already present for {filename}")
            if 'TIME_BARY' not in hdul[0].header:
                # Additional calculations based on header information
                data_exp = round(float(hdul[0].header['EXPTIME']), 2)
                half_exptime = data_exp / 2.
                time_isot = Time(hdul[0].header['DATE-OBS'], format='isot', scale='utc', location=get_location())
                time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
                time_jd += half_exptime * u.second
                ra = hdul[0].header['TELRAD']
                dec = hdul[0].header['TELDECD']
                ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
                time_bary = time_jd.tdb + ltt_bary
                time_helio = time_jd.utc + ltt_helio

                # Update the header with barycentric and heliocentric times
                hdul[0].header['TIME_BARY'] = (time_bary.jd, 'Barycentric Time (JD)')
                hdul[0].header['TIME_HELIO'] = (time_helio.jd, 'Heliocentric Time (JD)')
            else:
                print(f"TIME_BARY already present for {filename}")

                hdul.flush()
    print("All headers updated")


def main():
    """
    Main function for the script
    """

    parser = argparse.ArgumentParser(description='Add headers to FITS files')
    parser.add_argument('--directory', type=str, help='Path to the directory containing FITS files')
    args = parser.parse_args()

    if args.directory:
        custom_directory = args.directory
    else:
        custom_directory = os.getcwd()  # Use the current working directory if no custom directory is provided

    update_header(custom_directory)
    print(f"Using directory: {custom_directory}")


if __name__ == "__main__":
    main()

