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
        basename = os.path.basename(filename)
        prefix = basename[:11]
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


def delete_png_files(filenames):
    """
    Delete files starting with 'evening' or 'morning'.

    Parameters
    ----------
    filenames : list of str
        List of filenames to check and delete if they start with 'evening' or 'morning'.
    """
    for filename in filenames:
        basename = os.path.basename(filename)
        if basename.endswith('.png'):
            os.remove(filename)
            print(f"Deleted file: {filename}")


def main(directory):
    filenames = get_fits_filenames(directory)
    if not filenames:
        print("No .fits files found in the specified directory.")
        return

    delete_flat_files(filenames)

    delete_png_files(filenames)

    filtered_filenames = filter_files(filenames)
    if not filtered_filenames:
        print("No files to delete.")
        return

    prefixes = get_prefix(filtered_filenames)
    print('The prefixes are:', prefixes)

    for prefix in prefixes:
        prefix_filenames = [filename for filename in filenames if filename.startswith(prefix)]
        filtered_filenames_prefixes = filter_files(prefix_filenames)
        if filtered_filenames_prefixes:
            filtered_filenames_prefixes.sort()  # Sort the filenames
            delete_files(filtered_filenames_prefixes)
        else:
            print(f"No files to delete for prefix: {prefix}")

    print("All files have been deleted successfully.")


if __name__ == "__main__":
    directory = '.'
    main(directory)
