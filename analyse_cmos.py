#! /usr/bin/env python
import datetime
import json
import os
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from astropy.table import Table


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
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


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
        if filename.startswith(f"phot_{prefix}") and filename.endswith('.fits'):
            phot_files.append(filename)
    return phot_files


def read_phot_file(filename, prefix):
    """
    Read the photometry file.

    Parameters
    ----------
    filename : str
        Photometry file to read.
    prefix : str
        Prefix for the photometry file.

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
    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Filter filenames based on specific criteria
    filenames = filter_filenames(current_night_directory)
    print(f"Number of files: {len(filenames)}")

    # Extract unique prefixes from the filenames
    prefixes = get_prefix(filenames)
    print(f"The prefixes are: {prefixes}")

    # Process each prefix
    for prefix in prefixes:
        phot_files = get_phot_files(current_night_directory, prefix)
        print(f"Photometry files for {prefix}: {phot_files}")
        for phot_file in phot_files:
            phot_tab = read_phot_file(phot_file, prefix)
            if phot_tab is not None:
                print(f"Photometry table for {prefix}: {phot_tab}")


if __name__ == "__main__":
    main()
