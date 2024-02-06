#!/usr/bin/env python3
import glob
import os
from datetime import datetime, timedelta
from astropy.io import fits
import numpy as np
import shutil


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

    if os.path.exists(os.path.join(out_path, f'master_flat_{os.path.basename(current_night_directory)}.fits')):
        print('Found master flat in {}'.format(
            os.path.join(out_path, f'master_flat_{os.path.basename(current_night_directory)}.fits')))
        return fits.getdata(os.path.join(out_path, f'master_flat_{os.path.basename(current_night_directory)}.fits'))

    evening_files = [f for f in glob.glob(os.path.join(current_night_directory, 'evening*.fits')) if
                     'HDR' in fits.getheader(f)['READMODE']]

    if evening_files:
        files = evening_files
    else:
        # If evening files don't exist, use morning files
        files = [f for f in glob.glob(os.path.join(current_night_directory, 'morning*.fits')) if
                 'HDR' in fits.getheader(f)['READMODE']]

    if not files:
        print('No suitable flat field files found.')
        return None  # or handle the case where no files are found

    print('Creating master flat')
    # take only the first 21
    files = files[:21]

    cube = np.zeros((*master_bias.shape, len(files)))
    for i, f in enumerate(files):
        data, header = fits.getdata(f, header=True)
        cube[:, :, i] = data - master_bias - master_dark * header['EXPTIME'] / dark_exposure
        cube[:, :, i] = cube[:, :, i] / np.average(cube[:, :, i])

    master_flat = np.median(cube, axis=2)

    # Copy header from one of the input files
    header = fits.getheader(files[0])

    # Write the master flat with the copied header
    output_path = os.path.join(out_path, f'master_flat_{os.path.basename(current_night_directory)}.fits')
    hdu = fits.PrimaryHDU(master_flat, header=header)
    hdu.writeto(output_path, overwrite=True)

    hdul = fits.open(output_path, mode='update')
    hdul[0].header['FILTER'] = 'NGTS'
    hdul.close()

    return master_flat


def reduce_images(base_path, master_bias, master_dark, master_flat):
    current_night_directory = find_current_night_directory(base_path)
    if current_night_directory is None:
        print('Current night directory not found')
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

            # Save the reduced image with _r.fits suffix
            output_filename = os.path.join(current_night_directory,
                                           f"{os.path.splitext(os.path.basename(filename))[0]}_r.fits")
            fits.PrimaryHDU(fd, fh).writeto(output_filename, overwrite=True)


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def copy_master_files(master_bias_path, master_flat_path, master_dark_path, current_night_directory):
    # Copy master bias to the current night directory
    shutil.copy(master_bias_path, os.path.join(current_night_directory, 'master_bias.fits'))

    # Copy and rename master flat to the current night directory
    new_master_flat_path = os.path.join(current_night_directory, 'master_flat.fits')
    shutil.copy(master_flat_path, new_master_flat_path)

    # Copy master dark to the current night directory
    shutil.copy(master_dark_path, os.path.join(current_night_directory, 'master_dark.fits'))

    print('Master files copied to the current night directory.')
    return new_master_flat_path


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

    current_night_directory = find_current_night_directory(base_path)

    if current_night_directory:
        new_master_flat_path = copy_master_files(os.path.join(out_path, 'master_bias.fits'),
                                                 os.path.join(out_path,
                                                              f'master_flat_{os.path.basename(current_night_directory)}.fits'),
                                                 os.path.join(out_path, 'master_dark.fits'),
                                                 current_night_directory)

        # Rename the master flat file to "master_flat.fits"
        os.rename(new_master_flat_path, os.path.join(current_night_directory, 'master_flat.fits'))
    else:
        print('Current night directory not found')
