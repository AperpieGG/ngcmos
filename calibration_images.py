#!/usr/bin/env python
import glob
import os
from datetime import datetime, timedelta
from astropy.io import fits
import numpy as np


def bias(base_path, out_path):
    master_bias_path = os.path.join(out_path, 'master_bias.fits')

    if os.path.exists(master_bias_path):
        print('Found master bias in {}'.format(master_bias_path))
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
    master_dark_path = os.path.join(out_path, 'master_dark.fits')

    if os.path.exists(master_dark_path):
        print('Found master dark in {}'.format(master_dark_path))
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


def flat(base_path, out_path, master_bias, master_dark, dark_exposure=10):
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

    # If current_night_directory is None or the master flat for the current night directory doesn't exist
    # and current_night_directory is the current working directory
    if current_night_directory == os.getcwd():
        print('Current night directory is the current working directory.')
        master_flat_filename = 'master_flat.fits'
        master_flat_path = os.path.join(out_path, master_flat_filename)
        if os.path.exists(master_flat_path):
            print(f'Found master flat in {master_flat_path}')
            return fits.getdata(master_flat_path)
        else:
            print("Master flat file not found in out path:", master_flat_path)
            return None
    else:
        print('No current night directory found.')
        return None


def reduce_images(base_path, master_bias, master_dark, master_flat):
    current_night_directory = find_current_night_directory(base_path)
    if current_night_directory is None:
        current_night_directory = os.getcwd()
    else:
        print('Current night directory found {} will reduce images'.format(current_night_directory))

    for filename in sorted(glob.glob(os.path.join(current_night_directory, '*.fits'))):
        exclude = ['bias', 'dark', 'flat', 'evening', 'morning', '_r']
        if any([e in filename for e in exclude]):
            continue
        try:
            fd, fh = fits.getdata(filename, header=True)
            fd = (fd - master_bias - master_dark * fh['EXPTIME'] / 10) / master_flat
            fd_data_uint = fd.astype('uint16')
            limits = np.iinfo(fd_data_uint.dtype)
            fd_data_uint[fd < limits.min] = limits.min
            fd_data_uint[fd > limits.max] = limits.max
            fd = fd_data_uint
        except Exception as e:
            print(f'Failed to process {filename}. Exception: {str(e)}')
            continue

        print(f'Processed {filename}')


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    # First directories
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

    # Create the output directory if it doesn't exist
    create_directory_if_not_exists(out_path)

    master_bias = bias(calibration_path, out_path)
    master_dark = dark(calibration_path, out_path, master_bias)
    master_flat = flat(base_path, out_path, master_bias, master_dark)
    reduce_images(base_path, master_bias, master_dark, master_flat)
