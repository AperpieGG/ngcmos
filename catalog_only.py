#!/usr/bin/env python
"""
Run through many reference images, generate catalogs
and try solving them one by one (Adjusted to only generate catalogs)
"""
import os
import glob as g
import argparse as ap
from astropy.io import fits

# pylint: disable=invalid-name
# pylint: disable=no-member


def arg_parse():
    """
    Parse the command line arguments
    """
    p = ap.ArgumentParser("Solve AG references images for CASUTools")
    p.add_argument('--defocus',
                   help='manual override for defocus (mm)',
                   type=float,
                   default=0.0)
    p.add_argument('--force3rd',
                   help='force a 3rd order distortion polyfit',
                   action='store_true',
                   default=False)
    p.add_argument('--save_matched_cat',
                   help='output the matched catalog with basic photometry',
                   action='store_true',
                   default=False)
    return p.parse_args()


if __name__ == "__main__":
    # grab command line args
    args = arg_parse()

    # get a list of all fits images
    all_fits = sorted(g.glob("*.fits"))

    # filter out the catalogs
    ref_images = [f for f in all_fits if "_cat" not in f]

    # Extract prefixes from the OBJECT keyword in the headers
    prefixes = set()
    for fits_file in all_fits:
        exclude_list = ["bias", "dark", "flat", "morning", "evening"]
        if any(ex in fits_file for ex in exclude_list):
            continue  # Skip files containing excluded terms in their names
        with fits.open(fits_file) as hdulist:
            object_keyword = hdulist[0].header.get('OBJECT', '')
            prefix = object_keyword[:11]  # Take first 11 letters
            if prefix:  # Check if prefix is not empty
                prefixes.add(prefix)

    print("Found the following prefixes {} in the fits files\n".format(prefixes))

    # Iterate over prefixes
    for prefix in prefixes:
        # Find the first image for this prefix
        ref_image = next((img for img in ref_images if img.startswith(prefix)), None)
        if ref_image is None:
            print("No reference image found for prefix:", prefix)
            continue

        print("Found ref image {} with prefix: {}\n".format(ref_image, prefix))
        base_name = ref_image.split('.fits')[0]
        cat_file = f"{prefix}_catalog.fits"

        # Get the coordinates from the header to make a catalog
        with fits.open(ref_image) as ff:
            ra = str(ff[0].header['TELRAD'])
            dec = str(ff[0].header['TELDECD'])
            epoch = str(ff[0].header['DATE-OBS'])
            box_size = "2.8"  # You can adjust this as needed

        # Call the catalog maker only for the first image of each prefix
        if not os.path.exists(cat_file):
            cmd_args = ["/home/ops/refcatpipe2/cmos/make_ref_catalog.py",
                        ra, dec, box_size, box_size, epoch, cat_file]
            cmd = " ".join(cmd_args)
            os.system(cmd)
            print("Catalog created for image {} with prefix: {}\n".format(ref_image, prefix))
        else:
            print("Catalog already exists for image {} with prefix: {}\n".format(ref_image, prefix))
