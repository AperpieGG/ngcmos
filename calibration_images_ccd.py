#!/usr/bin/env python
"""
Find and read bias images, trim them and median combine them

"""

import os
from astropy.io import fits
import numpy as np


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
        if filename.endswith('.fits') and not filename.endswith('.fits.bz2'):
            fits_path = os.path.join(directory, filename)
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                if 'IMGTYPE' in header and header['IMGTYPE'] == 'BIAS':
                    filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def bias(directory):
    """
    Create the master bias from the bias files.

    Parameters
    ----------
    directory
    containing the bias files.

    Returns
    -------
    numpy.ndarray
        Master bias.
    """
    master_bias_path = os.path.join(directory, 'master_bias.fits')

    if os.path.exists(master_bias_path):
        print('Found master bias')
        return fits.getdata(master_bias_path)
    else:
        print('Did not find master bias, creating....')

        # Find and read the bias for hdr mode
        files = filter_filenames(directory)

        # Limit the number of files to the first 21
        files = files[:21]
        print(f'Found {len(files)} with shape {fits.open(files[0])[0].data.shape}')

        # check if we have an overscan to remove
        print('Checking for overscan to remove')
        for frame in fits.open(files[0]):
            if frame.data.shape == (2088, 2048):
                frame_data = frame.data[20:2068, :].astype(float)
                print('The frame shape is:', frame_data.shape)
            else:
                print(f"Invalid frame shape {frame.data.shape}")
                return None

        cube = np.zeros((2048, 2048, len(files)))
        for i, f in enumerate(files):
            cube[:, :, i] = fits.getdata(f)
        master_bias = np.median(cube, axis=2)

        # Copy header from one of the input files
        header = fits.getheader(files[0])

        fits.PrimaryHDU(master_bias, header=header).writeto(master_bias_path, overwrite=True)

        # zip the files apart from master_bias
        for filename in filenames:
            if filename != 'master_bias.fits':
                os.system(f"bzip2 {directory}/*.fits")
                print(f"Zipped files in {directory}")

        return master_bias


def reduce_image(directory, filenames):
    """

    Reduce the image by subtracting the master bias.

    """

    master_bias = bias(directory)

    if master_bias is None:
        return print('Master bias not found')

    # Find and read the image
    reduced_data = []

    for filename in filenames:
        try:
            fd, hdr = fits.getdata(filename, header=True)

            # Additional calculations based on header information

            # Reduce image
            fd = (fd - master_bias)
            reduced_data.append(fd)  # Append the reduced image to the list

            # Append the filename to the filenames list
            filenames.append(os.path.basename(filename))

        except Exception as e:
            print(f'Failed to process {filename}. Exception: {str(e)}')
            continue


if __name__ == '__main__':
    # Set the directory containing the bias files
    parent_directory = os.getcwd()

    # get a list of subdirectories inside the parent directory
    subdirectories = [name for name in os.listdir(parent_directory) if
                      os.path.isdir(os.path.join(parent_directory, name))]

    print('Action found:', subdirectories)

    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_biasFrames"):
            # form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # set directory for the current subdirectory
            directory = subdirectory_path
            print(f"Directory: {directory}")

            # unzip the files
            os.system(f"bzip2 -d {directory}/*.bz2")
            print(f"Unzipped files in {directory}")

            # Get the list of filenames
            filenames = filter_filenames(directory)
            print(f"Number of files: {len(filenames)}")

            # Reduce the image
            master_bias = bias(directory)





