#!/usr/bin/env python
import glob
import os
from astropy.io import fits
import numpy as np
from utils import find_current_night_directory

calibration_paths = '/home/ops/data/20231212/'
base_paths = '/home/ops/data/'
out_paths = '/home/ops/data/calibration_images/'

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def bias(out_path):
    """
    Create the master bias from the bias files.

    Parameters
    ----------
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
        print('Could not find Master bias, exiting!')
        return None


def dark(out_path):
    """
    Create the master dark from the dark files.

    Parameters
    ----------
    out_path : str
        Path to the output directory.

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
        print('Could not find Master dark, exiting!')
        return None


def flat(base_path, master_bias, master_dark, dark_exposure=10):
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
    # Find and read the flat files
    flat_files = glob.glob(os.path.join(base_path, 'evening*.fits'))

    # Limit the number of files to the first 21
    flat_files = flat_files[:21]

    # Read and stack the flat frames
    cube = np.zeros((*master_bias.shape, len(flat_files)))
    for i, f in enumerate(flat_files):
        data, header = fits.getdata(f, header=True)
        cube[:, :, i] = (data - master_bias - master_dark * header['EXPTIME'] / dark_exposure) / np.median(data)

    # Create the master flat by taking the median along the stack
    master_flat = np.median(cube, axis=2)

    # Save the master flat to the current working directory
    master_flat_filename = 'master_flat.fits'
    master_flat_path = os.path.join(os.getcwd(), master_flat_filename)
    fits.writeto(master_flat_path, master_flat, overwrite=True)

    print(f'Master flat saved to: {master_flat_path}')
    return master_flat


def main():
    """
    Main function to create the master bias, dark, and flat.
    """
    base_path = find_current_night_directory(base_paths[0])
    print(f'Using base path: {base_path}')

    master_bias = bias(out_path)
    master_dark = dark(out_path)
    flat(base_path, master_bias, master_dark)


if __name__ == '__main__':
    main()
