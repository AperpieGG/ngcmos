#!/usr/bin/env python

"""
This script grabs the diagnostics plots, shifts plots, and the mp4 movies
from the previous night.
"""

import os
from datetime import datetime, timedelta

# Define the primary and fallback paths
primary_path = '/home/ops/data/shifts_plots'
fallback_path = '/home/u5500483/shifts_plots_cmos'
output_file_path = '/home/ops/ngcmos/files_to_download.txt'


def find_last_night():
    # Get the previous date in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    return previous_date


def get_files(path, night):
    files = os.listdir(path)
    if files:
        files = [f for f in files if f.endswith(night + '.mp4') or f.endswith(night + '.pdf')]
    return files


def save_file_list(file_list, output_path):
    with open(output_path, "w") as f:
        for file in file_list:
            f.write("{}\n".format(file))


if __name__ == "__main__":
    # Check if primary path exists, otherwise use fallback path
    if os.path.exists(primary_path):
        path = primary_path
    else:
        print(f"Primary path '{primary_path}' does not exist. Using ngtshead path.")
        path = fallback_path

    last_night = find_last_night()
    files = get_files(path, last_night)
    print(f"Current night directory: {last_night}")

    # Print files without the path
    print(f"Files: {[os.path.basename(f) for f in files]}")

    # Get the absolute paths for saving to the file list
    files_with_paths = [os.path.join(path, f) for f in files]

    if files_with_paths:
        save_file_list(files_with_paths, output_file_path)
        print(f"File list saved to: {output_file_path}")
    else:
        print("No files found for the previous night.")