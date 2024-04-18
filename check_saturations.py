#!/usr/bin/env python
import os
import glob
import numpy as np
from astropy.io import fits
import sep


def identify_saturated_images(directory):
    """
    Identify images with saturated pixels in the given directory.

    Parameters:
    directory (str): Path to the directory containing FITS images.

    Returns:
    list: List of filenames with saturated pixels.
    """
    saturated_images = []

    # Get a list of FITS files in the directory
    fits_files = glob.glob(os.path.join(directory, '*.fits'))

    for fits_file in fits_files:
        # Open the FITS file
        hdulist = fits.open(fits_file)
        data = hdulist[0].data

        # Check for saturated pixels
        if np.any(data >= 65535):
            saturated_images.append(fits_file)

        hdulist.close()

    return saturated_images


if __name__ == "__main__":
    directory = "."  # Set the directory to the current working directory
    saturated_images = identify_saturated_images(directory)

    if saturated_images:
        print("Images with saturated pixels:")
        for image in saturated_images:
            print(image)
    else:
        print("No images with saturated pixels found.")
