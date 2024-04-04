#!/usr/bin/env python3
"""
Run through many reference images, generate catalogs
and try solving them one by one
"""
import os
import glob as g
from astropy.io import fits

# pylint: disable=invalid-name
# pylint: disable=no-member


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
    # get the current working directory
    parent_directory = os.getcwd()

    # get a list of subdirectories inside the parent directory
    subdirectories = get_subdirectories(parent_directory)

    print('The subdirectories are:', subdirectories)

    # iterate over each subdirectory
    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_observeField"):
            # form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # change the working directory to the subdirectory
            os.chdir(subdirectory_path)

            # Maintain a dictionary to keep track of which catalogs have been created
            catalog_dict = {}

            print(f"Processing subdirectory: {subdirectory_path}")

            # get a list of all fits images
            all_fits = sorted([f for f in g.glob("*.fits.bz2") if fits.getheader(f)['IMGCLASS'] == 'SCIENCE'])
            print("The number of science FITS files found:", len(all_fits))
            print("The prefix is:", fits.getheader(all_fits[0])['OBJECT'])

            # filter out the catalogs
            ref_images = [f for f in all_fits if "_cat" not in f]

            # Iterate over reference images
            for ref_image in ref_images:
                print("Processing reference image:", ref_image)
                base_name = ref_image.split('.fits.bz2')[0]
                cat_file = f"{base_name}_catalog.fits"

                # get the coords from the header, use this to make a catalog
                with fits.open(ref_image) as ff:
                    ra = str(ff[0].header['RA'])
                    dec = str(ff[0].header['DEC'])
                    epoch = str(ff[0].header['DATE-OBS'])
                    box_size = "2.8"  # You can adjust this as needed

                # Check if catalog for this prefix has already been created
                prefix = ref_image.split('_')[0]  # Extract prefix from filename
                if prefix not in catalog_dict:
                    catalog_dict[prefix] = cat_file  # Add catalog filename to dictionary
                    # Call the catalog maker only if it hasn't been created already
                    cmd_args = ["/home/ops/refcatpipe2/cmos/make_ref_catalog.py",
                                ra, dec, box_size, box_size, epoch, cat_file]
                    cmd = " ".join(cmd_args)
                    os.system(cmd)
                    print("Catalog created for reference image:", ref_image)
                else:
                    # Catalog for this prefix has already been created, use the existing one
                    cat_file = catalog_dict[prefix]

                if os.path.exists(cat_file):
                    print("Found catalog file:", cat_file)
                    print("Solving reference image:", ref_image)
                    # solve the reference image with this catalog file
                    cmd2_args = ["/home/ops/refcatpipe2/cmos/solve_ref_images.py",
                                 cat_file, ref_image]

                    cmd2 = " ".join(cmd2_args)
                    os.system(cmd2)
                    print("Solved reference image:", ref_image)

                # Now, iterate over all FITS files again and solve for each image
                for fits_file in all_fits:
                    if fits_file != ref_image and "_cat" not in fits_file:  # Exclude the reference image and catalogs
                        print("Processing FITS file:", fits_file)
                        with fits.open(fits_file) as hdulist:
                            # Check if keywords exist
                            if 'CTYPE1' not in hdulist[0].header or 'CTYPE2' not in hdulist[0].header:
                                print("Solving FITS file:", fits_file)
                                # solve the image with the same catalog file
                                cmd2_args = ["/home/ops/refcatpipe2/cmos/solve_ref_images.py",
                                             cat_file, fits_file]

                                cmd2 = " ".join(cmd2_args)
                                os.system(cmd2)
                                print("Solved FITS file:", fits_file)
                            else:
                                print("FITS file is already solved. Skipping:", fits_file)

            # move back to the parent directory for the next subdirectory iteration
            os.chdir(parent_directory)
