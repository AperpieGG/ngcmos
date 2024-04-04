#!/usr/bin/env python3
"""
Run through many reference images, generate catalogs
and try solving them one by one
"""
import os
import glob as g
from astropy.io import fits
import argparse as ap

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


def get_subdirectories(parent_directory):
    """
    Get a list of subdirectories inside the parent directory.

    Parameters
    ----------
    parent_directory : str
        The parent directory to search for subdirectories.

    Returns
    -------
    list of str
        List of subdirectories.
    """
    subdirectories = [name for name in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, name))]
    return subdirectories


if __name__ == "__main__":
    # Parse command-line arguments
    args = arg_parse()

    # Get the current working directory
    parent_directory = os.getcwd()

    # Get a list of subdirectories inside the parent directory
    subdirectories = [name for name in os.listdir(parent_directory) if
                      os.path.isdir(os.path.join(parent_directory, name))]

    print('The subdirectories are:', subdirectories)

    # Iterate over each subdirectory
    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_observeField"):
            # Form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # Change the working directory to the subdirectory
            os.chdir(subdirectory_path)

            print(f"Processing subdirectory: {subdirectory_path}")

            # Get a list of all FITS images
            all_fits = sorted([f for f in g.glob("*.fits.bz2") if fits.getheader(f)['IMGCLASS'] == 'SCIENCE'])
            print("The number of science FITS files found:", len(all_fits))

            # Check if there are any FITS files in the subdirectory
            if not all_fits:
                print("No science FITS files found in this subdirectory.")
                continue

            # Use the first FITS file to extract prefix and create catalog filename
            prefix = subdirectory  # Use subdirectory name as prefix
            cat_file = f"{prefix}_catalog.fits"

            # Check if catalog for this subdirectory has already been created
            if not os.path.exists(cat_file):
                # Get the coords from the header of the first FITS file
                with fits.open(all_fits[0]) as ff:
                    ra = str(ff[0].header['CMD_RA'])
                    dec = str(ff[0].header['CMD_DEC'])
                    epoch = str(ff[0].header['DATE-OBS'])
                    box_size = "2.8"  # You can adjust this as needed

                # Call the catalog maker
                cmd_args = ["/home/ops/refcatpipe2/cmos/make_ref_catalog.py",
                            ra, dec, box_size, box_size, epoch, cat_file]

                cmd = " ".join(cmd_args)
                os.system(cmd)
                print("Catalog created for subdirectory:", subdirectory)

            # Now, iterate over all FITS files and process each image
            for fits_file in all_fits:
                print("Processing FITS file:", fits_file)
                with fits.open(fits_file) as hdulist:
                    # Check if keywords exist
                    if 'CTYPE1' not in hdulist[0].header or 'CTYPE2' not in hdulist[0].header:
                        print("Solving FITS file:", fits_file)
                        # Solve the image with the catalog file
                        cmd2_args = ["/home/ops/refcatpipe2/cmos/solve_ref_images.py",
                                     cat_file, fits_file]

                        # Add optional arguments based on command-line arguments
                        if args.save_matched_cat:
                            cmd2_args.append("--save_matched_cat")
                        if args.defocus is not None:
                            cmd2_args.append("--defocus {:.2f}".format(args.defocus))
                        if args.force3rd:
                            cmd2_args.append("--force3rd")

                        cmd2 = " ".join(cmd2_args)
                        os.system(cmd2)
                        print("Solved FITS file:", fits_file)
                    else:
                        print("FITS file is already solved. Skipping:", fits_file)

            # Move back to the parent directory for the next subdirectory iteration
            os.chdir(parent_directory)