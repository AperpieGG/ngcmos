#!/usr/bin/env python

"""
This script is used to reduce the images in the specified directory.
It will create a master bias or read it if it already exists in the calibration directory.
It will create a master dark or read it if it already exists in the calibration directory.

If this script works as a cronjob and the night directory is found then it will check if the
master_flat_<night_directory>.fits already exists in the calibration path and use that.
Otherwise, it will create it and use it for the reduction of the images.

If the current night directory is not found (running it manually) then it will create
a master_flat.fits (created from the create_flats.py) from the flat files in the
current working directory and use it for the reduction of the images.

if the master_flat is not created from the create_flats then it will take the general master_flat.fits
from the calibration directory and use it for the reduction of the images.
"""

import glob
import os
from datetime import datetime, timedelta
from astropy.io import fits
import numpy as np
from astropy.time import Time
import astropy.units as u
from utils import get_location, get_light_travel_times


def bias(base_path, out_path):
    """
    Create the master bias from the bias files.

    Parameters
    ----------
    base_path : str
        Base path for the directory.
    out_path : str
        Path to the output directory.

    Returns
    -------
    numpy.ndarray
        Master bias.
    """
    master_bias_path = os.path.join(out_path, 'master_bias.fits')

    if os.path.exists(master_bias_path):
        print('Found master bias')
        return fits.getdata(master_bias_path)
    else:
        print('Creating master bias')

        # Find and read the bias for hdr mode
        files = [f for f in glob.glob(os.path.join(base_path, 'bias*.fits')) if 'HDR' in fits.getheader(f)['READMODE']]

        # Limit the number of files to the first 21
        files = files[:21]

        cube = np.zeros((2048, 2048, len(files)))
        for i, f in enumerate(files):
            cube[:, :, i] = fits.getdata(f)
        master_bias = np.median(cube, axis=2)

        # Copy header from one of the input files
        header = fits.getheader(files[0])

        fits.PrimaryHDU(master_bias, header=header).writeto(master_bias_path, overwrite=True)
        return master_bias


def dark(base_path, out_path, master_bias):
    """
    Create the master dark from the dark files.

    Parameters
    ----------
    base_path : str
        Base path for the directory.
    out_path : str
        Path to the output directory.
    master_bias : numpy.ndarray
        Master bias.

    Returns
    -------
    numpy.ndarray
        Master dark.
    """
    master_dark_path = os.path.join(out_path, 'master_dark.fits')

    if os.path.exists(master_dark_path):
        print('Found master dark')
        return fits.getdata(master_dark_path)
    else:
        print('Creating master dark')

        # Find and read the darks for hdr mode
        files = [f for f in glob.glob(os.path.join(base_path, 'dark*.fits')) if 'HDR' in fits.getheader(f)['READMODE']]

        # Limit the number of files to the first 21
        files = files[:21]

        cube = np.zeros((2048, 2048, len(files)))
        for i, f in enumerate(files):
            cube[:, :, i] = fits.getdata(f)
        master_dark = np.median(cube, axis=2) - master_bias

        # Copy header from one of the input files
        header = fits.getheader(files[0])

        fits.PrimaryHDU(master_dark, header=header).writeto(master_dark_path, overwrite=True)
        return master_dark


def find_current_night_directory(base_path):
    """
    Find the directory for the current night based on the current date.
    if not then use the current working directory.

    Parameters
    ----------
    base_path : str
        Base path for the directory.

    Returns
    -------
    str or None
        Path to the current night directory if found, otherwise None.
    """

    # Get the previous date directory in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(base_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        # Use the current working directory
        return None


# lines 172-222 may be removed cause this script does not run automatically anymore
# TODO: remove the conditions for the find current night directory and only use the flats
#  from the create flats
def flat(base_path, out_path, master_bias, master_dark, dark_exposure=10):
    """
    Create the master flat from the flat files.

    Parameters
    ----------
    base_path : str
        Base path for the directory.
    out_path : str
        Path to the output directory.
    master_bias : numpy.ndarray
        Master bias.
    master_dark : numpy.ndarray
        Master dark.
    dark_exposure : int
        Dark exposure time.

    Returns
    -------
    numpy.ndarray
        Master flat.
    """
    current_night_directory = find_current_night_directory(base_path)

    # If current_night_directory is None, set it to the current working directory
    if current_night_directory is None:
        current_night_directory = os.getcwd()
    elif current_night_directory:  # Check if current_night_directory is not None and has a value
        # Check if there is a master flat specific to the current night directory
        current_night_master_flat_filename = f'master_flat_{os.path.basename(current_night_directory)}.fits'
        current_night_master_flat_path = os.path.join(out_path, current_night_master_flat_filename)

        if os.path.exists(current_night_master_flat_path):
            print(f'Found master flat for current night directory in {current_night_master_flat_path}')
            return fits.getdata(current_night_master_flat_path)

        # If the master flat for the current night directory doesn't exist, create it
        print(f'Master flat for current night directory not found. Creating...')

        # Find appropriate files for creating the master flat
        evening_files = [f for f in glob.glob(os.path.join(current_night_directory, 'evening*.fits')) if
                         'HDR' in fits.getheader(f)['READMODE']]

        if not evening_files:
            # If evening files don't exist, use morning files
            evening_files = [f for f in glob.glob(os.path.join(current_night_directory, 'morning*.fits')) if
                             'HDR' in fits.getheader(f)['READMODE']]

        if not evening_files:
            print('No suitable flat field files found.')
            return None  # or handle the case where no files are found

        print('Creating master flat')
        # take only the first 21
        files = evening_files[:21]

        cube = np.zeros((*master_bias.shape, len(files)))
        for i, f in enumerate(files):
            data, header = fits.getdata(f, header=True)
            cube[:, :, i] = data - master_bias - master_dark * header['EXPTIME'] / dark_exposure
            cube[:, :, i] = cube[:, :, i] / np.average(cube[:, :, i])

        master_flat = np.median(cube, axis=2)

        # Copy header from one of the input files
        header = fits.getheader(files[0])

        # Write the master flat with the copied header
        hdu = fits.PrimaryHDU(master_flat, header=header)
        hdu.writeto(current_night_master_flat_path, overwrite=True)

        hdul = fits.open(current_night_master_flat_path, mode='update')
        hdul[0].header['FILTER'] = 'NGTS'
        hdul.close()
        print(f'Master flat for current night directory created in {current_night_master_flat_path}')
        return master_flat

    if current_night_directory == os.getcwd():
        print('Current night directory is the current working directory.')

        if os.path.exists(os.path.join(current_night_directory, 'master_flat.fits')):
            print('Using current working directory and the master flat found in:',
                  os.path.join(current_night_directory, 'master_flat.fits'))
            return fits.getdata(os.path.join(current_night_directory, 'master_flat.fits'))

        elif os.path.exists(os.path.join(out_path, 'master_flat.fits')):
            print('Using master flat found in:', os.path.join(out_path, 'master_flat.fits'))
            return fits.getdata(os.path.join(out_path, 'master_flat.fits'))
        else:
            print("Master flat file not found in out path:", os.path.join(out_path, 'master_flat.fits'))
            return None
    else:
        print('No current night directory found.')
        return None


def reduce_images(base_path, out_path, prefix_filenames):
    """
    Reduce the images in the specified directory.

    Parameters
    ----------
    base_path : str
        Base path for the directory.
    out_path : str
        Path to the output directory.
    prefix_filenames : list of str
        List of filenames for the prefix.

    Returns
    -------
    tuple
        Tuple containing reduced data, reduced header information, and filenames.
    """
    master_bias = bias(base_path, out_path)
    master_dark = dark(base_path, out_path, master_bias)
    master_flat = flat(base_path, out_path, master_bias, master_dark)

    reduced_data = []
    reduced_header_info = []
    filenames = []

    for filename in prefix_filenames:
        try:
            fd, hdr = fits.getdata(filename, header=True)

            # Additional calculations based on header information
            data_exp = round(float(hdr['EXPTIME']), 2)
            half_exptime = data_exp / 2.
            time_isot = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += half_exptime * u.second
            ra = hdr['TELRAD']
            dec = hdr['TELDECD']
            ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
            time_bary = time_jd.tdb + ltt_bary
            time_helio = time_jd.utc + ltt_helio

            # Reduce image
            fd = (fd - master_bias - master_dark * hdr['EXPTIME'] / 10) / master_flat
            reduced_data.append(fd)  # Append the reduced image to the list
            reduced_header_info.append(hdr)

            # Append the filename to the filenames list
            filenames.append(os.path.basename(filename))

        except Exception as e:
            print(f'Failed to process {filename}. Exception: {str(e)}')
            continue

        print(f'Reduced {filename}')

    return reduced_data, reduced_header_info, filenames


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
