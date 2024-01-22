import glob
import os
from datetime import datetime, timedelta
from astropy.io import fits
import numpy as np
import re


def bias(base_path, out_path):
    master_bias_path = os.path.join(out_path, 'master_bias.fits')

    if os.path.exists(master_bias_path):
        print('Master bias already exists')
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
        fits.PrimaryHDU(master_bias).writeto(master_bias_path, overwrite=True)
        return master_bias


def dark(base_path, out_path, master_bias):
    master_dark_path = os.path.join(out_path, 'master_dark.fits')

    if os.path.exists(master_dark_path):
        print('Master dark already exists')
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
        fits.PrimaryHDU(master_dark).writeto(master_dark_path, overwrite=True)
        return master_dark


def find_current_night_directory(base_path):
    # Get the previous date directory in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(base_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return None


def flat(base_path, out_path, master_bias, master_dark, dark_exposure=10):
    current_night_directory = find_current_night_directory(base_path)

    if current_night_directory is None:
        print('Current night directory not found')
        return

    if os.path.exists(os.path.join(out_path, f'master_flat_{os.path.basename(current_night_directory)}.fits')):
        print('Master flat already exists')
        return

    files = [f for f in glob.glob(os.path.join(current_night_directory, 'evening*.fits')) if
             'HDR' in fits.getheader(f)['READMODE']]
    if not files:
        print('No HDR flat files found.')
        return

    print('Creating master flat')
    # take only the first 21
    files = files[:21]

    cube = np.zeros((*master_bias.shape, len(files)))
    for i, f in enumerate(files):
        data, header = fits.getdata(f, header=True)
        cube[:, :, i] = data - master_bias - master_dark * header['EXPTIME'] / dark_exposure
        cube[:, :, i] = cube[:, :, i] / np.average(cube[:, :, i])

    master_flat = np.median(cube, axis=2)
    fits.PrimaryHDU(master_flat).writeto(
        os.path.join(out_path, f'master_flat_{os.path.basename(current_night_directory)}.fits'), overwrite=True)
    del cube
    del master_flat


def flat_up_one_directory(calibration_path, out_path, master_bias, master_dark, dark_exposure=10):
    # Get the parent directory of the calibration_path
    flat_path = os.path.abspath(os.path.join(calibration_path, os.pardir))

    # Call the flat function with the modified flat_path
    flat(flat_path, out_path, master_bias, master_dark, dark_exposure)


if __name__ == '__main__':
    calibration_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/20231212/'
    out_path = '/Users/u5500483/Documents/GitHub/ngcmos/'  # to be changed for nuc (home/ops/calibration_images)
    master_bias = bias(calibration_path, out_path)
    master_dark = dark(calibration_path, out_path, master_bias)
    flat_up_one_directory(calibration_path, out_path, master_bias, master_dark)
