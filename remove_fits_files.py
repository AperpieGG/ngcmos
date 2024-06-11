#! /usr/bin/env python
import os
import glob
from astropy.io import fits


def get_fits_filenames(directory):
    """
    Get a list of all .fits files in the specified directory.

    Parameters
    ----------
    directory : str
        Directory to search for .fits files.

    Returns
    -------
    list of str
        List of .fits filenames.
    """
    return glob.glob(os.path.join(directory, "*.fits"))


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


def filter_files(filenames):
    """
    Filter filenames by the shape of the data in the .fits files and ignore specific words.

    Parameters
    ----------
    filenames : list of str
        List of .fits filenames.
    Returns
    -------
    list of str
        List of filenames with data of the specified shape, excluding files with specific words.
    """
    filtered_filenames = []
    for filename in filenames:
        # Check if the filename contains any of the ignored words
        if any(word in filename.lower() for word in ['master', 'flat', 'catalog', 'phot', 'morning', 'evening']):
            continue
    return filtered_filenames


def delete_files(filenames):
    """
    Delete all files in the list except the first one.

    Parameters
    ----------
    filenames : list of str
        List of filenames to be deleted, except the first one.
    """
    for filename in filenames[1:]:
        os.remove(filename)
        print(f"Deleted file: {filename}")


def delete_flat_files(filenames):
    """
    Delete files starting with 'evening' or 'morning'.

    Parameters
    ----------
    filenames : list of str
        List of filenames to check and delete if they start with 'evening' or 'morning'.
    """
    for filename in filenames:
        basename = os.path.basename(filename)
        if basename.startswith('evening') or basename.startswith('morning'):
            os.remove(filename)
            print(f"Deleted file: {filename}")


def main(directory):
    # Step 1: Get all .fits filenames
    filenames = get_fits_filenames(directory)
    if not filenames:
        print("No .fits files found in the specified directory.")
        return

    delete_flat_files(filenames)
    # Step 2: Extract unique prefixes
    prefixes = get_prefix(filenames)

    # Step 3: For each prefix, filter by shape and delete files
    for prefix in prefixes:
        prefix_filenames = [filename for filename in filenames if filename.startswith(prefix)]
        filtered_filenames = filter_files(prefix_filenames)
        if filtered_filenames:
            filtered_filenames.sort()  # Sort the filenames
            delete_files(filtered_filenames)
        else:
            print(f"No files to delete for prefix: {prefix}")

    print("All files have been deleted successfully.")


if __name__ == "__main__":
    directory = '.'
    main(directory)
