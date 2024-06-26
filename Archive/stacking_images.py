#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np
from astropy.io import fits
from utils import utc_to_jd

TODO = """
1. Grab images in a list
2. sort the list (in terms of name/time
3. create an array to split depending on the number of images to be stacked (1 min binned)
4. stack the images
5. take the jd_mid from the first image and the last image and average them to get the mid point of the observation
6. save the stacked image in the out directory with the name of the first image stacked_{first_image}.fits
7. save the jd_mid on the header of the stacked image.
"""

parser = argparse.ArgumentParser(description="Stack FITS images.")
parser.add_argument("n_images_to_stack", type=int, help="Number of images to stack together.")
args = parser.parse_args()

# Grab images in a list
images = glob.glob("*.fits")
exclude = ['stacked', 'master_flat', 'master_dark', 'master_bias', 'catalog']
images.sort()

# create an array to split depending on the number of images 422 images to be stacked (1 min binned)
n_images = len(images)
n_images_to_stack = args.n_images_to_stack
n_splits = n_images // n_images_to_stack
image_splits = np.array_split(images, n_splits)

# stack the images
for i, image_split in enumerate(image_splits):
    # convert DATE-OBS to JD
    jd_start = utc_to_jd(fits.getheader(image_split[0])['DATE-OBS'])
    jd_end = utc_to_jd(fits.getheader(image_split[-1])['DATE-OBS'])
    # take the jd_mid from the first image and the last image and average them to get the mid point of the observation
    jd_mid = (float(jd_start) + float(jd_end)) / 2
    # save the stacked image in the out directory with the name of the first image stacked_{first_image}.fits
    stacked_image = fits.HDUList()
    for j, image in enumerate(image_split):
        if j == 0:
            stacked_image = fits.open(image)
            stacked_image[0].data = stacked_image[0].data.astype(float)
        else:
            # convert the data to float32 to avoid overflow
            stacked_image[0].data += fits.open(image)[0].data.astype(float)
    stacked_image_filename = f"stacked_{image_split[0]}"
    stacked_image[0].header['JD-MID'] = jd_mid
    stacked_image.writeto(stacked_image_filename)
    stacked_image.close()
    print(f"Stacked image saved as {stacked_image_filename} with JD-MID = {jd_mid}")
