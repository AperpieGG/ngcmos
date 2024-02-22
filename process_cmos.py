#! /usr/bin/env python
import os
from datetime import datetime, timedelta
from collections import defaultdict
from calibration_images import reduce_images
from donuts import Donuts
import numpy as np
from utils import catalogue_to_pixels, parse_region_content
import json
import warnings
from astropy.io import fits

# ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)


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
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def check_headers(directory, filenames):
    """
    Check headers of all files for CTYPE1 and CTYPE2.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    no_wcs = os.path.join(directory, 'no_wcs')
    if not os.path.exists(no_wcs):
        os.makedirs(no_wcs)

    for file in filenames:
        file_path = os.path.join(directory, file)

        try:
            with fits.open(file_path) as hdulist:
                header = hdulist[0].header
                ctype1 = header.get('CTYPE1')
                ctype2 = header.get('CTYPE2')

                if ctype1 is None or ctype2 is None:
                    print(f"Warning: {file} does not have CTYPE1 and/or CTYPE2 in the header. Moving to "
                          f"'no_wcs' directory.")
                    new_path = os.path.join(no_wcs, file)
                    os.rename(file_path, new_path)

        except Exception as e:
            print(f"Error checking header for {file}: {e}")

    print("Done checking headers, number of files without CTYPE1 and/or CTYPE2:", len(os.listdir(no_wcs)))


def check_donuts(filenames):
    """
    Check donuts for each group of images.

    Parameters
    ----------
    filenames : list of filenames.

    """
    grouped_filenames = defaultdict(list)
    for filename in filenames:
        prefix = get_prefix(filename)
        grouped_filenames[prefix].append(filename)

    for prefix, filenames in grouped_filenames.items():
        filenames.sort()
        reference_image = filenames[0]
        d = Donuts(reference_image)
        for filename in filenames[1:]:
            shift = d.measure_shift(filename)
            sx = round(shift.x.value, 2)
            sy = round(shift.y.value, 2)
            print(f'{filename} shift X: {sx} Y: {sy}')
            shifts = np.array([abs(sx), abs(sy)])
            if np.sum(shifts > 50) > 0:
                print(f'{filename} image shift too big X: {sx} Y: {sy}')
                if not os.path.exists('failed_donuts'):
                    os.mkdir('failed_donuts')
                comm = f'mv {filename} failed_donuts/'
                print(comm)
                os.system(comm)


def get_prefix(filename):
    """
    Extract prefix from filename
    """
    return filename[:11]


def find_first_image_of_each_prefix(filenames):
    first_images = {}
    for filename in sorted(filenames):
        prefix = get_prefix(filename)
        if prefix not in first_images:
            first_images[prefix] = filename
    return first_images


def get_region_files(filenames):
    grouped_region_files = defaultdict(set)
    for filename in filenames:
        prefix = get_prefix(filename)
        exclude_keywords = ['catalog', 'morning', 'evening', 'bias', 'flat', 'dark']
        if any(keyword in filename for keyword in exclude_keywords):
            continue
        region_files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith('_input.reg')]
        grouped_region_files[prefix].update(region_files)
    return grouped_region_files


def read_region_files(region_files):
    region_contents = {}
    for region_file in region_files:
        with open(region_file, 'r') as file:
            contents = file.read()
            region_contents[region_file] = contents
    return region_contents


def main():
    # set directory for the current night or use the current working directory
    directory = find_current_night_directory(base_path)

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames)

    # Check donuts for each group
    check_donuts(filenames)

    # Calibrate images and get FITS files
    reduced_data, jd_list, bjd_list, hjd_list, filenames = reduce_images(base_path, out_path)

    # Get region files for each prefix
    region_files = get_region_files(filenames)

    # Read the contents of region files
    region_contents = {}
    for prefix, files in region_files.items():
        region_contents[prefix] = read_region_files(files)

    for prefix, contents in region_contents.items():
        first_images = find_first_image_of_each_prefix(filenames)
        for region_file, region_content in contents.items():
            ra_dec_coords = parse_region_content(region_content)
            print(f"Prefix: {prefix}, Region File: {region_file}, coordinates: {ra_dec_coords}")

            if prefix in first_images:
                xy_coordinates = catalogue_to_pixels(first_images[prefix], ra_dec_coords)
                print("X coordinates:", xy_coordinates[0])
                print("Y coordinates:", xy_coordinates[1])
            else:
                print(f"No image found for prefix {prefix}")


if __name__ == "__main__":
    main()


# TODO: Script working from path/to/data but not from random directory