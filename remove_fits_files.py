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

def filter_filenames(filenames):
    """
    Filter filenames to exclude those containing specific words.

    Parameters
    ----------
    filenames : list of str
        List of .fits filenames.

    Returns
    -------
    list of str
        List of filenames excluding those with specific words.
    """
    filtered_filenames = []
    for filename in filenames:
        # Check if the filename contains any of the ignored words
        if any(word in filename.lower() for word in ['catalog', 'phot', 'rel', 'master', 'morning', 'evening']):
            continue

        filtered_filenames.append(filename)

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


def delete_png_files(directory):
    """
    Delete all .png files in the specified directory.

    Parameters
    ----------
    directory : str
        Directory to search for .png files.
    """
    png_files = glob.glob(os.path.join(directory, "*.png"))
    for filename in png_files:
        os.remove(filename)
        print(f"Deleted file: {filename}")


def main(directory):
    # Step 1: Get all .fits filenames
    filenames = get_fits_filenames(directory)
    if not filenames:
        print("No .fits files found in the specified directory.")
        return

    # Step 2: Filter filenames
    filtered_filenames = filter_filenames(filenames)

    # Step 3: Extract unique prefixes from filtered filenames
    prefixes = get_prefix(filtered_filenames)
    print(f'found the following prefixes: {prefixes}')

    # Step 4: For each prefix, delete files
    for prefix in prefixes:
        prefix_filenames = [filename for filename in filtered_filenames if filename.startswith(prefix)]
        if prefix_filenames:
            prefix_filenames.sort()  # Sort the filenames
            delete_files(prefix_filenames)
        else:
            print(f"No files to remove for prefix {prefix}")

    # Delete files starting with 'evening' or 'morning'
    delete_flat_files(filenames)

    # Delete all .png files
    delete_png_files(directory)

if __name__ == "__main__":
    directory = '.'
    main(directory)
