#! /usr/bin/env python
import datetime
import json
import os
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits


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


def get_phot_files(directory, prefix):
    """
    Get photometry files from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        List of photometry files.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if filename.startswith('phot') and filename.endswith(prefix):
            phot_files.append(filename)
    return phot_files


def main():
    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Filter filenames based on specific criteria
    filenames = os.listdir(current_night_directory)

    # Extract unique prefixes from the filenames
    prefixes = get_prefix(filenames)

    # Process each prefix
    for prefix in prefixes:
        # Get photometry files for the current prefix
        phot_files = get_phot_files(current_night_directory, prefix)

        # Process photometry files
        for phot_file in phot_files:
            # Construct the full path to the photometry file
            phot_file_path = os.path.join(current_night_directory, phot_file)

            # Read the photometry file
            # You can read the photometry file here using fits or any other appropriate method

            # Example: Reading photometry file using fits
            try:
                with fits.open(phot_file_path) as ff:
                    # Access the data in the photometry file as needed
                    tab = ff[1].data
                    print(tab)
            except Exception as e:
                print(f"Error reading photometry file {phot_file}: {e}")


if __name__ == "__main__":
    main()