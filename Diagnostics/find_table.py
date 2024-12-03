#! /usr/bin/env python
import argparse
import os
from astropy.io import fits


def find_rows_in_fits(file_path, frame_id, tic_id):
    """
    Find rows in a FITS table based on specific `frame_id` and `tic_id` values.

    Parameters:
        file_path (str): Path to the FITS file.
        frame_id (str): The `frame_id` value to search for.
        tic_id (int): The `tic_id` value to search for.

    Returns:
        list: List of dictionaries with information for matching rows.
    """
    try:
        # Open the FITS file and load the data
        with fits.open(file_path) as hdul:
            data = hdul[1].data  # Assuming the table is in the first extension

        # Find rows matching both frame_id and tic_id
        mask = (data['frame_id'] == frame_id) & (data['tic_id'] == tic_id)
        matching_rows = data[mask]

        # Extract information for matching rows
        results = [dict(zip(data.names, row)) for row in matching_rows]

        return results

    except Exception as e:
        print(f"Error: {e}")
        return []


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
    phot_file = []
    for filename in os.listdir(directory):
        if filename.startswith('phot'):
            phot_file.append(filename)
    return phot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and plot RMS for two files.')
    parser.add_argument('frame_id', type=str, help='frame_id label')
    parser.add_argument('tic_id', type=int, help='tic_id label')
    args = parser.parse_args()

    # Path to your FITS file
    directory = '.'
    fits_filename = get_phot_files(directory)

    if not fits_filename:
        print("No photometry files found in the directory.")
        exit(1)
    else:
        print(f'Found photometry table: {fits_filename}, will read it...')

    # Call the function and retrieve results
    rows = find_rows_in_fits(fits_filename[0], args.frame_id, args.tic_id)

    # Print the results
    if rows:
        print("Matching rows found:")
        for i, row in enumerate(rows, start=1):
            print(f"Row {i}:")
            for key, value in row.items():
                print(f"  {key}: {value}")
            print("\n")
    else:
        print("No matching rows found.")