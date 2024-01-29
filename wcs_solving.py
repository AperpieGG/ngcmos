#!/usr/bin/env python3
import os
from datetime import datetime, timedelta
from astropy.io import fits


def solve_reference_image(refimage):
    """
    Solve the reference image to obtain better WCS

    Parameters
    ----------
    refimage : string
        path to the reference image for solving

    Returns
    -------
    solved : boolean
        Did the image solve ok?

    Raises
    ------
    None
    """
    solved_refimage = f"{os.path.splitext(refimage)[0]}.new"

    if not os.path.exists(solved_refimage):
        with fits.open(refimage) as hdulist:
            header = hdulist[0].header
            ra = float(header['TELRAD'])
            dec = float(header['TELDECD'])

        # Build the astrometry.net command
        command = (
            # f"/opt/homebrew/bin/solve-field {refimage} " # for full path on Mac
            f"solve-field {refimage} "
            f"--ra {ra:.6f} --dec {dec:.6f} "
            "--radius 5 "
            "--overwrite "
            "--no-plots --no-verify -z 2 --tweak-order 3 --cpulimit 600"
        )

        try:
            os.system(command)
        except Exception as e:
            print(f"Error solving image: {e}")
            return False

        # Remove the original .fits file
        try:
            os.remove(refimage)
            print(f"Removed original file: {refimage}")
        except Exception as e:
            print(f"Error removing original file {refimage}: {e}")

        # Rename the solved image to use .fits instead of .new
        os.rename(solved_refimage, f"{os.path.splitext(refimage)[0]}.fits")
        print(f"Renamed solved image to: {os.path.splitext(refimage)[0]}.fits")

    return os.path.exists(f"{os.path.splitext(refimage)[0]}.fits")


def solve_all_images_in_directory(directory):
    """
    Solve WCS for all FITS images in the specified directory

    Parameters
    ----------
    directory : string
        path to the directory containing FITS images

    Returns
    -------
    None
    """
    exclude_words = ["evening", "morning", "flat", "bias", "dark"]

    # Get a sorted list of FITS files in the directory
    fits_files = sorted([filename for filename in os.listdir(directory) if filename.endswith(".fits")
                         and not any(word in filename.lower() for word in exclude_words)])

    for filename in fits_files:
        filepath = os.path.join(directory, filename)
        if solve_reference_image(filepath):
            print(f"WCS solved successfully for {filename}")
        else:
            print(f"Failed to solve WCS for {filename}")


def remove_unwanted_files(directory):
    unwanted_extensions = ['.xyls', '.axy', '.corr', '.match', '.rdls', '.solved', '.wcs']

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in unwanted_extensions):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Removed unwanted file: {filename}")
            except Exception as e:
                print(f"Error removing file {filename}: {e}")


def find_current_night_directory(file_path):
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


def main():
    file_path = "/Users/u5500483/Downloads/DATA_MAC/CMOS/"

    current_night_directory = find_current_night_directory(file_path)

    if current_night_directory:
        print(f"Current night directory found: {current_night_directory}")
        solve_all_images_in_directory(current_night_directory)
        remove_unwanted_files(current_night_directory)
    else:
        print("No current night directory found.")


if __name__ == "__main__":
    main()
