#! /usr/bin/env python
import datetime
import json
import os
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt


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


def plot_first_gaia_id_vs_jd_mid(table):
    # Get unique frame_ids
    unique_frame_ids = set(table['frame_id'])

    for frame_id in unique_frame_ids:
        # Select rows corresponding to the current frame_id
        mask = table['frame_id'] == frame_id
        frame_data = table[mask]

        # Get the first gaia_id and first jd_mid
        first_flux_2 = frame_data['flux_2'][0]
        first_jd_mid = frame_data['jd_mid'][0]
        first_gaia_id = frame_data['gaia_id'][0]

        # Plot first_gaia_id vs first_jd_mid
        plt.scatter(first_jd_mid, first_flux_2, label=first_gaia_id)

    # Add labels and legend
    plt.xlabel('JD Mid')
    plt.ylabel('First Gaia ID')
    plt.title('First Gaia ID vs JD Mid for Each Frame')
    plt.show()


def main():
    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Filter filenames based on specific criteria
    filenames = filter_filenames(current_night_directory)
    print(f"Number of files: {len(filenames)}")

    # Extract unique prefixes from the filenames
    prefixes = get_prefix(filenames)
    print(f"The prefixes are: {prefixes}")

    first_prefix = next(iter(prefixes), None)
    if first_prefix is not None:
        phot_files = get_phot_files(current_night_directory, first_prefix)
        print(f"Photometry files for {first_prefix}: {phot_files}")
        for phot_file in phot_files:
            phot_tab = read_phot_file(phot_file, first_prefix)
            if phot_tab is not None:
                print('Plotting...')
                plot_first_gaia_id_vs_jd_mid(phot_tab)


if __name__ == "__main__":
    main()
