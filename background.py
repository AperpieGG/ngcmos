#!/usr/bin/env python
import argparse
import datetime
import json
import os
import fnmatch
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from utils import plot_images


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


def get_image_data(frame_id, image_directory):
    """
    Get the image data corresponding to the given frame_id.

    Parameters:
        frame_id (str): The frame_id of the image.
        image_directory (str): The directory where the image files are stored.

    Returns:
        numpy.ndarray or None: The image data if the image exists, otherwise None.
    """
    # Construct the path to the image file using the frame_id
    image_path = os.path.join(image_directory, frame_id)

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


def find_background_level(table, gaia_id_to_plot, exposure_time=10, image_directory=""):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]

    # Get the image data for the specified frame_id
    for frame_id in gaia_id_data['frame_id']:
        image_data = get_image_data(frame_id, image_directory)
        sky_level = np.median(image_data)

    Total_sky = np.median(sky_level)
    return Total_sky


def get_phot_files(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'phot_*.fits'):
            phot_files.append(os.path.join(directory, filename))
    return phot_files


def read_phot_file(filename):
    """
    Read the photometry file.

    Parameters
    ----------
    filename : str
        Photometry file to read.

    Returns
    -------
    astropy.table.table.Table
        Table containing the photometry data.
    """
    # Read the photometry file here using fits or any other appropriate method
    try:
        with fits.open(filename) as ff:
            # Access the data in the photometry file as needed
            tab = ff[1].data
            return tab
    except Exception as e:
        print(f"Error reading photometry file {filename}: {e}")
        return None


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Find the background level for a specified Gaia ID.')
    args = parser.parse_args()

    # Find the directory for the current night
    current_night_directory = find_current_night_directory(base_path)

    # Get the photometry files for the current night
    phot_files = get_phot_files(current_night_directory)

    # Read the photometry data from the table
    phot_data = []
    for phot_file in phot_files:
        phot_data.append(read_phot_file(phot_file))

    # Find the background level for the specified Gaia ID
    background_level = find_background_level(phot_data, current_night_directory)
    print('Final background level is:', background_level)


if __name__ == "__main__":
    main()




