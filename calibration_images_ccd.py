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


def trim_bias(directory):
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

    # Find and read the bias for hdr mode
    files = filter_filenames(directory)

    # join the directory path to the filenames
    files_filtered = [os.path.join(directory, f) for f in files]
    print(f'Found {len(files_filtered)} bias files in {directory}')

    if fits.open(files_filtered[0])[0].data.shape == (2048, 2048):
        print('The files are already trimmed')
    else:
        # check if we have an overscan to remove
        for filename in files_filtered:
            frame = fits.open(filename)
            frame = frame[0]
            print('Initial frame shape:', frame.data.shape)
            if frame.data.shape == (2048, 2088):  # overscan present: x,y
                frame_data = frame.data[:, 20:2068].astype(float)
                print('Final frame shape:', frame_data.shape)
                frame = fits.PrimaryHDU(frame_data.astype(np.uint16), header=frame.header)
                frame.writeto(filename, overwrite=True)

        print('Trimmed bias files to shape:', frame_data.shape)


def bias(directory):

    master_bias_path = os.path.join(directory, 'master_bias.fits')

    if os.path.exists(master_bias_path):
        print('Found master bias')
        return fits.getdata(master_bias_path)
    else:
        print('Did not find master bias, creating....')

        # Find and read the bias for hdr mode
        files = filter_filenames(directory)

        # Limit the number of files to the first 21
        files_filtered = files[:21]

        # join the directory path to the filenames
        files_filtered = [os.path.join(directory, f) for f in files_filtered]
        print(f'Found {len(files_filtered)} bias files in {directory}')

        try:
            # create a 3D cube of the bias files
            cube = np.zeros((2048, 2048, len(files_filtered)))
            for i, f in enumerate(files_filtered):
                cube[:, :, i] = fits.getdata(f)

            master_bias = np.median(cube, axis=2)

            header = fits.getheader(files_filtered[0])
            fits.PrimaryHDU(master_bias, header=header).writeto(master_bias_path, overwrite=True)

        except Exception as e:
            print(f'Failed to create master bias. Exception: {str(e)}')
            return None

        # zip the files apart from master_bias
        for filename in files:
            if filename != 'master_bias.fits':
                os.system(f"bzip2 {os.path.join(directory, filename)}")
                print(f"Zipped file: {filename}")

        print('Master bias created and stored in:', master_bias_path)
        return master_bias


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

            # trim the bias images (2048, 2088) to (2048, 2048)
            trim_bias(directory)

            # Reduce the image
            bias(directory)





