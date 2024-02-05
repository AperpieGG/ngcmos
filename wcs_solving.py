#!/usr/bin/env python3

"""
This script solves the WCS for all FITS images in the specified directory
and removes unwanted files.

Usage:
python wcs_solving.py [--directory <your_directory>] all
python wcs_solving.py [--directory <your_directory>] first

The first command will solve the WCS for all FITS images in the specified or current directory
and remove unwanted files. The second command will solve the WCS for the first FITS image
"""
import argparse
import os
from datetime import datetime, timedelta
from astropy.io import fits


def find_current_night_directory(file_path):
    """
    Find the directory for the current night based on the current date.

    Parameters
    ----------
    file_path : str
        Base path for the directory.

    Returns
    -------
    str or None
        Path to the current night directory if found, otherwise None.
    """

    # Get the current date in the format YYYYMMDD
    current_date = datetime.now().strftime("%Y%m%d") + '/'
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d") + '/'

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(file_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return None


def solve_reference_image(refimage):
    """
    Solve the reference image to obtain better WCS if not already present.

    Parameters
    ----------
    refimage : str
        Path to the reference image for solving.

    Returns
    -------
    bool
        True if the image was solved successfully, False otherwise.
    """
    with fits.open(refimage) as hdulist:
        header = hdulist[0].header

        # Check if WCS information is already present
        if 'CTYPE1' not in header or 'CTYPE2' not in header:
            log_file = f"{refimage.split('.fits')[0]}.astrom_log"
            solved_refimage = f"{os.path.splitext(refimage)[0]}.new"

            # Build the astrometry.net command
            command = (
                f"solve-field {refimage} "
                f"--ra {header['TELRAD']:.6f} --dec {header['TELDECD']:.6f} "
                "--radius 5 "
                # "--overwrite "
                "--scale-units arcsecperpix \
                 --scale-low 3.9 --scale-high 4.3"
                "--skip-solved --no-plots --no-verify -z 2 --tweak-order 3 --cpulimit 600 --continue"
            )

            try:
                os.system(command)
                return True  # Image solved successfully
            except Exception as e:
                print(f"Error solving image: {e}")
                return False
        else:
            print(f"WCS already present in header for {refimage}")
            return True


def solve_all_images_in_directory(directory):
    """
    Solve WCS for all FITS images in the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing FITS images.
    """
    exclude_words = ["evening", "morning", "flat", "bias", "dark"]

    # Get a sorted list of FITS files in the directory
    fits_files = sorted([filename for filename in os.listdir(directory) if filename.endswith(".fits")
                         and not any(word in filename.lower() for word in exclude_words)])

    for filename in fits_files:
        filepath = os.path.join(directory, filename)
        solve_reference_image(filepath)


def remove_unwanted_files(directory):
    """
    Remove unwanted files and rename .new files to .fits.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    unwanted_extensions = ['.xyls', '.axy', '.corr', '.match', '.rdls', '.solved', '.wcs']

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in unwanted_extensions):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {filename}: {e}")

    for filename in os.listdir(directory):
        if filename.endswith('.new'):
            file_path = os.path.join(directory, filename)
            try:
                os.rename(file_path, f"{os.path.splitext(file_path)[0]}.fits")
            except Exception as e:
                print(f"Error renaming file {filename}: {e}")


def check_headers(directory):
    """
    Check headers of all FITS files for CTYPE1 and CTYPE2.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    no_wcs = os.path.join(directory, 'no_wcs')

    # Create 'no_wcs' subdirectory if it doesn't exist
    if not os.path.exists(no_wcs):
        os.makedirs(no_wcs)

    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            exclude_words = ["evening", "morning", "flat", "bias", "dark"]
            if any(word in filename.lower() for word in exclude_words):
                continue

            file_path = os.path.join(directory, filename)

            try:
                with fits.open(file_path) as hdulist:
                    header = hdulist[0].header
                    ctype1 = header.get('CTYPE1')
                    ctype2 = header.get('CTYPE2')

                    if ctype1 is None or ctype2 is None:
                        print(f"Warning: {filename} does not have CTYPE1 and/or CTYPE2 in the header. Moving to "
                              f"'no_wcs' directory.")
                        # Move the file to 'no_wcs' subdirectory
                        new_path = os.path.join(no_wcs, filename)
                        os.rename(file_path, new_path)

            except Exception as e:
                print(f"Error checking header for {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Solve WCS for FITS images and remove unwanted files.")
    parser.add_argument("command", choices=["all", "first"], help="Choose 'all' or 'first' command.")
    parser.add_argument("--directory", help="Specify the directory. If not provided, the current night directory will "
                                            "be used.")

    args = parser.parse_args()

    if args.command == 'all':
        if args.directory:
            custom_directory = args.directory
            print(f"Custom directory provided: {custom_directory}")
        else:
            custom_directory = find_current_night_directory("/Users/u5500483/Downloads/DATA_MAC/CMOS/")
            if custom_directory:
                print(f"Current night directory found: {custom_directory}")
            else:
                print("No current night directory found.")
                return

        solve_all_images_in_directory(custom_directory)
        remove_unwanted_files(custom_directory)
        check_headers(custom_directory)

    elif args.command == 'first':
        if args.directory:
            custom_directory = args.directory
            print(f"Custom directory provided: {custom_directory}")
        else:
            custom_directory = find_current_night_directory("/Users/u5500483/Downloads/DATA_MAC/CMOS/")
            if custom_directory:
                print(f"Current night directory found: {custom_directory}")
            else:
                print("No current night directory found.")
                return

        first_image = os.path.join(custom_directory, next(
            (filename for filename in os.listdir(custom_directory) if filename.endswith(".fits")), None))
        solve_reference_image(first_image)
        remove_unwanted_files(custom_directory)

    else:
        print("Invalid argument. Use 'all' or 'first'.")


if __name__ == "__main__":
    main()

