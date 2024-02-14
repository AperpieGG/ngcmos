#! /usr/bin/env python
import os
from datetime import datetime, timedelta
import pyregion


# First directory
base_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/'
# Second directory
base_path_2 = '/home/ops/data/'

# Check if the first directory exists
if os.path.exists(base_path_1):
    base_path = base_path_1
else:
    base_path = base_path_2


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    if not then use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str or None
        Path to the current night directory if found, otherwise None.
    """

    # Get the previous date directory in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(directory, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        # Use the current working directory
        return os.getcwd()


def filter_region(directory):
    """
    Filter the region files in the given directory, keeping only the shapes with color=green.

    Parameters
    ----------
    directory : str
        Path to the directory containing the region files.

    Returns
    -------
    list of str
        List of paths to the filtered region files.
    """
    filtered_files = []

    # Get all the region files in the directory
    region_files = [f for f in os.listdir(directory) if f.endswith('_master.reg')]
    if region_files:
        # Extract unique prefixes from region file names
        unique_prefixes = {f[:11] for f in region_files}
        for prefix in unique_prefixes:
            filtered_files.extend(filter_region_with_prefix(directory, prefix))

    return filtered_files


def filter_region_with_prefix(directory, prefix):
    """
    Filter the region files in the given directory with the specified prefix,
    keeping only the shapes with color=green.

    Parameters
    ----------
    directory : str
        Path to the directory containing the region files.
    prefix : str
        Prefix to match against region file names.

    Returns
    -------
    list of str
        List of paths to the filtered region files.
    """
    prefix_filtered_files = []

    region_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('_master.reg')]
    if region_files:
        for region_file_name in region_files:
            region_file_path = os.path.join(directory, region_file_name)
            try:
                region_shapes = pyregion.open(region_file_path)
                # Filter shapes with color=green
                filtered_shapes = [shape for shape in region_shapes if 'green' in shape.attr]
                # Create a new region file path for the filtered shapes
                filtered_region_file_path = region_file_path.replace('_master.reg', '_filtered.reg')
                # Write the filtered shapes to the new region file
                with open(filtered_region_file_path, 'w') as f:
                    for shape in filtered_shapes:
                        f.write(str(shape) + '\n')
                prefix_filtered_files.append(filtered_region_file_path)
            except Exception as e:
                print(f"Error filtering region file {region_file_path}: {e}")

    return prefix_filtered_files


def main():
    # Get the current night directory or use the current working directory
    current_night_directory = find_current_night_directory(base_path)
    if current_night_directory:
        # Find all unique prefixes from region file names
        region_files = [f for f in os.listdir(current_night_directory) if f.endswith('_master.reg')]
        prefixes = set(f[:11] for f in region_files)

        if prefixes:
            for prefix in prefixes:
                filtered_files = filter_region_with_prefix(current_night_directory, prefix)
                if filtered_files:
                    print(f"Filtered region files with prefix '{prefix}' saved to:")
                    for filtered_file in filtered_files:
                        print(filtered_file)
                else:
                    print(f"No region files found with prefix '{prefix}' or error occurred during filtering.")
        else:
            print("No region files found in the directory.")
    else:
        print("Current night directory not found.")


if __name__ == "__main__":
    main()