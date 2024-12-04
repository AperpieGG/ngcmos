#!/usr/bin/env python
import numpy as np
from astropy.io import fits
import os


def get_phot_file(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the file.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    return [os.path.join(directory, filename) for filename in os.listdir(directory)
            if filename.startswith('phot') and filename.endswith('.fits')]


def get_catalog_file(directory):
    """
    Get catalog files with the pattern 'catalog.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the file.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    return [os.path.join(directory, filename) for filename in os.listdir(directory)
            if filename.endswith('catalog.fits')]


def main():

    catalog = get_catalog_file('.')
    phot_file = get_phot_file('.')

    with fits.open(catalog, mode='update') as hdul:
        catalog_data = hdul[1].data  # Assuming the table is in the first extension

    with fits.open(phot_file, mode='update') as hdul:
        phot_data = hdul[1].data

    print(f'Read data  all good {catalog_data}, {phot_data}')


if __name__ == "__main__":
    main()