#! /usr/bin/env python
import os
from datetime import datetime, timedelta
from collections import defaultdict
from calibration_images import reduce_images
from donuts import Donuts
import numpy as np
from utils import catalogue_to_pixels, parse_region_content
import json
import warnings
from astropy.io import fits
from astropy.table import Table, hstack
import fitsio
import sep
from astropy.wcs import WCS

# ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)


GAIN = 1.0
MAX_ALLOWED_PIXEL_SHIFT = 50
N_OBJECTS_LIMIT = 200
APERTURE_RADII = [2, 3, 4, 5, 6]
RSI = 15
RSO = 20
DEFOCUS = 0.0
AREA_MIN = 10
AREA_MAX = 200
SCALE_MIN = 4.5
SCALE_MAX = 5.5
DETECTION_SIGMA = 3
ZP_CLIP_SIGMA = 3


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    If not found, use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str
        Path to the current night directory.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    return current_date_directory if os.path.isdir(current_date_directory) else os.getcwd()


def filter_filenames(directory):
    """
    Filter filenames based on specific criteria.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        Filtered list of filenames.
    """
    filtered_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def get_prefix(filenames):
    """
    Extract prefix from filename
    """
    return filenames[:11]


def check_headers(directory, filenames):
    """
    Check headers of all files for CTYPE1 and CTYPE2.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    no_wcs = os.path.join(directory, 'no_wcs')
    if not os.path.exists(no_wcs):
        os.makedirs(no_wcs)

    for file in filenames:
        try:
            with fits.open(os.path.join(directory, file)) as hdulist:
                header = hdulist[0].header
                ctype1 = header.get('CTYPE1')
                ctype2 = header.get('CTYPE2')

                if ctype1 is None or ctype2 is None:
                    print(f"Warning: {file} does not have CTYPE1 and/or CTYPE2 in the header. Moving to "
                          f"'no_wcs' directory.")
                    new_path = os.path.join(no_wcs, file)
                    os.rename(os.path.join(directory, file), new_path)

        except Exception as e:
            print(f"Error checking header for {file}: {e}")

    print("Done checking headers, number of files without CTYPE1 and/or CTYPE2:", len(os.listdir(no_wcs)))


def check_donuts(filenames):
    """
    Check donuts for each group of images.

    Parameters
    ----------
    filenames : list of filenames.

    """
    grouped_filenames = defaultdict(list)
    for filename in filenames:
        prefix = get_prefix(filename)
        grouped_filenames[prefix].append(filename)

    for prefix, filenames in grouped_filenames.items():
        filenames.sort()
        reference_image = filenames[0]
        d = Donuts(reference_image)
        for filename in filenames[1:]:
            shift = d.measure_shift(filename)
            sx = round(shift.x.value, 2)
            sy = round(shift.y.value, 2)
            print(f'{filename} shift X: {sx} Y: {sy}')
            shifts = np.array([abs(sx), abs(sy)])
            if np.sum(shifts > 50) > 0:
                print(f'{filename} image shift too big X: {sx} Y: {sy}')
                if not os.path.exists('failed_donuts'):
                    os.mkdir('failed_donuts')
                comm = f'mv {filename} failed_donuts/'
                print(comm)
                os.system(comm)


def find_first_image_of_each_prefix(filenames):
    first_images = {}
    for filename in sorted(filenames):
        prefix = get_prefix(filename)
        if prefix not in first_images:
            first_images[prefix] = filename
    return first_images


def get_region_files(filenames):
    grouped_region_files = defaultdict(set)
    for filename in filenames:
        prefix = get_prefix(filename)
        exclude_keywords = ['catalog', 'morning', 'evening', 'bias', 'flat', 'dark']
        if any(keyword in filename for keyword in exclude_keywords):
            continue
        region_files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith('_input.reg')]
        grouped_region_files[prefix].update(region_files)
    return grouped_region_files


def read_region_files(region_files):
    region_contents = {}
    for region_file in region_files:
        with open(region_file, 'r') as file:
            contents = file.read()
            region_contents[region_file] = contents
    return region_contents


def get_catalog(filename, ext=0):
    """
    Read a fits image and header with fitsio

    Parameters
    ----------
    filename : str
        filename to load
    ext : int
        extension to load

    Returns
    -------
    data : array
        data from the corresponding extension
    header : fitsio.header.FITSHDR
        list from file header

    Raises
    ------
    None
    """
    data, header = fitsio.read(filename, header=True, ext=ext)
    return data, header


def convert_coords_to_pixels(cat, prefix, filenames):
    """
    Convert RA and DEC coordinates to pixel coordinates

    Parameters
    ----------
    data : array
        The image to search
    cat : astropy.table.table.Table
        Table containing the RA and DEC coordinates

    Returns
    -------
    x, y : float
        Pixel coordinates

    Raises
    ------
    None
    """
    # get the reference image
    cat = get_catalog()
    # get the WCS
    wcs_header = fits.getheader(find_first_image_of_each_prefix(filenames)[prefix])
    # convert the ra and dec to pixel coordinates
    x, y = WCS(wcs_header).all_world2pix(cat['ra_deg_corr'], cat['dec_deg_corr'], 1)
    return x, y


def find_max_pixel_value(data, x, y, radius):
    """
    Find the maximum pixel value in the image
    in a square around the aperture centre

    Parameters
    ----------
    data : array-like
        The image to search
    x : int
        X coordinate of the search box
    y : int
        Y coordinate of the search box
    radius : int
        The half width of the search box

    Returns
    -------
    max_pixel_value : int
        The maximum pixel value in the area provided

    Raises
    ------
    None
    """
    return round(data[int(y-radius):int(y+radius),
                      int(x-radius):int(x+radius)].ravel().max(), 2)


def wcs_phot(data, x, y, rsi, rso, aperture_radii, gain=1.12):
    """
    Take a corrected image array and extract photometry for a set of WCS driven
    X and Y pixel positons. Do this for a series of aperture radii and apply
    a gain correction to the photometry
    """
    col_labels = ["flux", "fluxerr", "flux_w_sky", "fluxerr_w_sky", "max_pixel_value"]
    Tout = None
    for r in aperture_radii:
        flux, fluxerr, _ = sep.sum_circle(data, x, y, r,
                                          subpix=0,
                                          bkgann=(rsi, rso),
                                          gain=gain)
        flux_w_sky, fluxerr_w_sky, _ = sep.sum_circle(data, x, y, r,
                                                      subpix=0,
                                                      gain=gain)
        # calculate the max pixel value in each aperture
        max_pixel_value = np.array([find_max_pixel_value(data, int(i), int(j), int(r+1)) for i, j in zip(x, y)])
        # build this photometry into a table
        if Tout is None:
            Tout = Table([flux, fluxerr, flux_w_sky, fluxerr_w_sky, max_pixel_value],
                          names=tuple([f"{c}_{r}" for c in col_labels]))
        else:
            T = Table([flux, fluxerr, flux_w_sky, fluxerr_w_sky, max_pixel_value],
                       names=tuple([f"{c}_{r}" for c in col_labels]))
            # stack the new columns onto the RHS of the table
            Tout = hstack([Tout, T])
    return Tout


def main():
    # set directory for the current night or use the current working directory
    directory = find_current_night_directory(base_path)

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)

    # get the prefix for the files
    prefix = get_prefix(filenames)
    print(f"Prefix: {prefix}")

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames)

    # Check donuts for each group
    check_donuts(filenames)

    # Calibrate images and get FITS files
    reduce_images(base_path, out_path)

    # get data from the catalog
    phot_cat = get_catalog(f"{base_path}/{prefix}_catalog_input.fits", ext=1)
    # convert the ra and dec to pixel coordinates
    phot_x, phot_y = convert_coords_to_pixels(phot_cat, prefix, filenames)
    print(f"X coordinates: {phot_x}")
    print(f"Y coordinates: {phot_y}")


    # # build output file preamble
    # frame_ids = [fitsfile for i in range(len(phot_x))]
    # frame_preamble = Table([frame_ids, phot_cat['gaia_id'], jd_list.value,
    #                         hjd_list.value, bjd_list.value, phot_x, phot_y],
    #                        names=("frame_id", "gaia_id", "jd_mid", "hjd_mid",
    #                               "bjd_mid", "x", "y"))
    #
    # # extract photometry at locations
    # frame_phot = wcs_phot(frame_data_corr, phot_x, phot_y, RSI, RSO, APERTURE_RADII, gain=1.12)
    #
    # # stack the phot and preamble
    # frame_output = hstack([frame_preamble, frame_phot])


if __name__ == "__main__":
    main()
