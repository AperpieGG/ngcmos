#!/usr/bin/env python
import argparse
import datetime
import json
import os
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from calibration_images import reduce_images


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def get_image_data(filename, image_directory):
    """
    Get the image data corresponding to the given frame_id.

    Parameters:
        filename (str): The name of the image.
        image_directory (str): The directory where the image files are stored.

    Returns:
        numpy.ndarray or None: The image data if the image exists, otherwise None.
    """
    # Construct the path to the image file using the frame_id
    image_path = os.path.join(image_directory, filename)

    # Check if the image file exists
    if os.path.exists(image_path):
        # Open the image file
        try:
            image_data = fits.getdata(image_path)
            return image_data
        except Exception as e:
            print(f"Error opening image file {image_path}: {e}")
            return None
    else:
        print(f"Image file {image_path} not found.")
        return None


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    If not found, use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str
        Path to the current night directory.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    return current_date_directory if os.path.isdir(current_date_directory) else os.getcwd()


def get_prefix(filenames):
    """
    Extract unique prefixes from a list of filenames.

    Parameters
    ----------
    filenames : list of str
        List of filenames.

    Returns
    -------
    set of str
        Set of unique prefixes extracted from the filenames.
    """
    prefixes = set()
    for filename in filenames:
        prefix = filename[:11]
        prefixes.add(prefix)
    return prefixes


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
        if filename.endswith('.fits'):
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot", "catalog_input"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def main():
    # set directory for the current night or use the current working directory
    directory = find_current_night_directory(base_path)
    print(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    print(f"Number of files: {len(filenames)}")

    # Get prefixes for each set of images
    prefixes = get_prefix(filenames)
    print(f"The prefixes are: {prefixes}")

    for prefix in prefixes:
        sky_list = []
        # Iterate over filenames with the current prefix
        prefix_filenames = [filename for filename in filenames if filename.startswith(prefix)]
        for filename in prefix_filenames:
            print(f"Processing filename {filename}......")
            # Calibrate image and get FITS file
            reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])
            sky_data = np.median(reduced_data)
            sky_list.append(sky_data)
            print(f"Sky data: {sky_data}")

        # Calculate the median sky value
        median_sky = np.median(sky_list)
        print(f"Median sky value for prefix {prefix} is: {median_sky}")


if __name__ == "__main__":
    main()
