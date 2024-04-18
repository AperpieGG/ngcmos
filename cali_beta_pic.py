#!/usr/bin/env python
import glob
import os
from datetime import datetime, timedelta
from astropy.io import fits
import numpy as np
from astropy.time import Time
import astropy.units as u
from utils import get_location, get_light_travel_times


def bias(out_path):
    master_bias_path = os.path.join(out_path, 'master_bias.fits')

    if os.path.exists(master_bias_path):
        print('Found master bias')
        return fits.getdata(master_bias_path)


def find_current_night_directory(base_path):
    """
    Find the directory for the current night based on the current date.
    if not then use the current working directory.

    Parameters
    ----------
    base_path : str
        Base path for the directory.

    Returns
    -------
    str or None
        Path to the current night directory if found, otherwise None.
    """

    # Get the previous date directory in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(base_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        # Use the current working directory
        return None


# lines 172-222 may be removed cause this script does not run automatically anymore
def flat(out_path):
    master_flat_path = os.path.join(out_path, 'master_flat.fits')

    if os.path.exists(master_flat_path):
        print('Found master bias')
        return fits.getdata(master_flat_path)


def reduce_images(out_path, prefix_filenames):
    master_bias = bias(out_path)
    master_flat = flat(out_path)

    reduced_data = []
    reduced_header_info = []
    filenames = []

    for filename in prefix_filenames:
        try:
            fd, hdr = fits.getdata(filename, header=True)

            # Additional calculations based on header information
            data_exp = round(float(hdr['EXPTIME']), 2)
            half_exptime = data_exp / 2.
            time_isot = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += half_exptime * u.second
            ra = hdr['TELRAD']
            dec = hdr['TELDECD']
            ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
            time_bary = time_jd.tdb + ltt_bary
            time_helio = time_jd.utc + ltt_helio

            # Reduce image
            fd = (fd - master_bias) / master_flat
            reduced_data.append(fd)  # Append the reduced image to the list
            reduced_header_info.append(hdr)

            # Append the filename to the filenames list
            filenames.append(os.path.basename(filename))

        except Exception as e:
            print(f'Failed to process {filename}. Exception: {str(e)}')
            continue

        print(f'Reduced {filename}')

    return reduced_data, reduced_header_info, filenames


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_reduced_images(reduced_data, reduced_header_info, filenames, out_path):
    """
    Save the reduced images to the specified directory.
    Parameters
    ----------
    reduced_data : list of numpy.ndarray
        List of reduced data.
    reduced_header_info : list of astropy.io.fits.header.Header
        List of reduced header information.
    filenames : list of str
        List of filenames.
    out_path : str
        Path to the output directory.
    """
    for data, header, filename in zip(reduced_data, reduced_header_info, filenames):
        out_filename = os.path.join(out_path, f'reduced_{filename}')
        # Check if the output file already exists, if yes, remove it before saving
        if os.path.exists(out_filename):
            os.remove(out_filename)
            print(f"Existing file {out_filename} removed.")
        fits.PrimaryHDU(data, header=header).writeto(out_filename, overwrite=True)
        print(f'Saved reduced image to {out_filename}')


def main():
    out_path = os.getcwd()

    # Get a list of filenames in the base path
    filenames = glob.glob(os.path.join(out_path, '*.fits'))

    exclude = ["catalog_input", "catalog", "master_dark", "master_bias", "master_flat"]

    # Filter the filenames
    filenames = [filename for filename in filenames if not any(word in filename.lower() for word in exclude)]

    # Iterate over the filenames
    for filename in filenames:
        print(f'Processing filename {filename}......')
        reduced_data, reduced_header, filenames = reduce_images(out_path, [filename])
        save_reduced_images(reduced_data, reduced_header, filenames, out_path)


if __name__ == '__main__':
    main()