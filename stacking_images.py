#!/usr/bin/env python
import os
from astropy.io import fits

# Get the current working directory
current_directory = os.getcwd()

# Get a list of all FITS files in the current directory
fits_files = [f for f in os.listdir(current_directory) if f.endswith('.fits')]

# Iterate over each FITS file
for fits_file in fits_files:
    # Open the FITS file and check if it has the 'JD-MID' keyword in its header
    with fits.open(os.path.join(current_directory, fits_file)) as hdul:
        header = hdul[0].header
        if 'JD-MID' not in header:
            # Delete the file if 'JD-MID' keyword is not found
            os.remove(os.path.join(current_directory, fits_file))
            print(f"Deleted {fits_file} because it does not have 'JD-MID' keyword.")
        else:
            print(f"Kept {fits_file} because it has 'JD-MID' keyword.")