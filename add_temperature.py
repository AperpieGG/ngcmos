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
    str
        Path to the photometry file.
    """
    for filename in os.listdir(directory):
        if filename.startswith('phot') and filename.endswith('.fits'):
            return os.path.join(directory, filename)
    raise FileNotFoundError("No photometry file found in the directory.")


def get_catalog_file(directory):
    """
    Get catalog files with the pattern 'catalog.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the file.

    Returns
    -------
    str
        Path to the catalog file.
    """
    for filename in os.listdir(directory):
        if filename.endswith('catalog.fits'):
            return os.path.join(directory, filename)
    raise FileNotFoundError("No catalog file found in the directory.")


def main():
    catalog_file = get_catalog_file('.')
    phot_file = get_phot_file('.')

    # Read the catalog file
    with fits.open(catalog_file) as catalog_hdul:
        catalog_data = catalog_hdul[1].data  # Assuming the table is in the first extension
        catalog_tic_ids = catalog_data['TIC_ID']
        catalog_teff = catalog_data['Teff']  # Assuming 'Teff' column exists

    # Read the photometry file
    with fits.open(phot_file, mode='update') as phot_hdul:
        phot_data = phot_hdul[1].data
        phot_tic_ids = phot_data['TIC_ID']  # Assuming 'TIC_ID' column exists

        # Add a new column for Teff if it doesn't exist
        if 'Teff' not in phot_data.names:
            new_col = np.zeros(len(phot_data), dtype=np.float32)  # Default value is 0
            phot_hdul[1].data = fits.BinTableHDU.from_columns(
                phot_hdul[1].columns + fits.Column(name='Teff', format='E', array=new_col)
            ).data

        # Update the Teff column in the photometry file
        for i, tic_id in enumerate(phot_tic_ids):
            if tic_id in catalog_tic_ids:
                idx = np.where(catalog_tic_ids == tic_id)[0][0]
                phot_data['Teff'][i] = catalog_teff[idx]
                print(f"Updated TIC_ID {tic_id}: Teff = {catalog_teff[idx]}")
            else:
                print(f"TIC_ID {tic_id} not found in the catalog.")

        # Save changes
        phot_hdul.flush()
        print("Teff information added to the photometry file.")


if __name__ == "__main__":
    main()