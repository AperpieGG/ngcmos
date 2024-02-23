#! /usr/bin/env python
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from calibration_images import reduce_images
from donuts import Donuts
import numpy as np
from astropy.coordinates import SkyCoord
import json
import warnings
from astropy.io import fits
from astropy.table import Table, hstack
import fitsio
import sep
from astropy import units as u
from astropy.wcs import WCS

# ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)

GAIN = 1.12
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

OK, TOO_FEW_OBJECTS, UNKNOWN = range(3)


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


def check_donuts(filenames, prefixes):
    """
    Check donuts for each group of images.

    Parameters
    ----------
    filenames : list of filenames.
    prefixes : list of prefixes.

    """
    for prefix in prefixes:
        group_filenames = [filename for filename in filenames if filename.startswith(prefix)]
        group_filenames.sort()

        reference_image = group_filenames[0]
        d = Donuts(reference_image)

        for filename in group_filenames[1:]:
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


def load_fits_image(filename, ext=0, force_float=True):
    """
    Read a fits image and header with fitsio

    Parameters
    ----------
    filename : str
        filename to load
    ext : int
        extension to load
    force_float : bool
        force image data to be floated on load

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
    if force_float:
        data = data.astype(float)
    return data, header


def convert_coords_to_pixels(cat, filenames):
    """
    Convert RA and DEC coordinates to pixel coordinates using the first image for each prefix.

    Parameters
    ----------
    cat : astropy.table.table.Table
        Catalog containing the RA and DEC coordinates
    filenames : list of str
        List of filenames corresponding to each prefix

    Returns
    -------
    x, y : numpy arrays
        Pixel coordinates

    Raises
    ------
    None
    """
    # Initialize lists to store pixel coordinates
    all_x = []
    all_y = []

    # Iterate over each set of filenames corresponding to each prefix
    for prefix_filenames in filenames:
        # Take the first filename for the current prefix
        filename = prefix_filenames[0]

        # Open the FITS file and extract the WCS information
        with fits.open(filename) as hdul:
            wcs_header = hdul[0].header

            # Convert the RA and DEC to pixel coordinates
            wcs = WCS(wcs_header)
            x, y = wcs.all_world2pix(cat['ra_deg_corr'], cat['dec_deg_corr'], 1)

            # Append the pixel coordinates to the lists
            all_x.append(x)
            all_y.append(y)

    # Convert lists to numpy arrays
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    return all_x, all_y


def _detect_objects_sep(data, background_rms, area_min, area_max,
                        detection_sigma, defocus_mm, trim_border=10):
    """
    Find objects in an image array using SEP

    Parameters
    ----------
    data : array
        Image array to source detect on
    background_rms
        Std of the sky background
    area_min : int
        Minimum number of pixels for an object to be valid
    area_max : int
        Maximum number of pixels for an object to be valid
    detection_sigma : float
        Number of sigma above the background for source detecting
    defocus_mm : float
        Level of defocus. Used to select kernel for source detect
    trim_border : int
        Number of pixels to exclude from the edge of the image array

    Returns
    -------
    objects : astropy Table
        A list of detected objects in astropy Table format

    Raises
    ------
    None
    """
    # set up some defocused kernels for sep
    kernel1 = np.array([[1, 1, 1, 1, 1],
                        [1, 2, 3, 2, 1],
                        [1, 3, 1, 3, 1],
                        [1, 2, 3, 2, 1],
                        [1, 1, 1, 1, 1]])
    kernel2 = np.array([[1, 1, 1, 1, 1, 1],
                        [1, 2, 3, 3, 2, 1],
                        [1, 3, 1, 1, 3, 1],
                        [1, 3, 1, 1, 3, 1],
                        [1, 2, 3, 3, 2, 1],
                        [1, 1, 1, 1, 1, 1]])
    kernel3 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 3, 3, 3, 3, 3, 3, 1],
                        [1, 3, 2, 2, 2, 2, 3, 1],
                        [1, 3, 2, 1, 1, 2, 3, 1],
                        [1, 3, 2, 1, 1, 2, 3, 1],
                        [1, 3, 2, 2, 2, 2, 3, 1],
                        [1, 3, 3, 3, 3, 3, 3, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1]])

    # check for defocus
    if defocus_mm >= 0.15 and defocus_mm < 0.3:
        print("Source detect using defocused kernel 1")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel1)
    elif defocus_mm >= 0.3 and defocus_mm < 0.5:
        print("Source detect using defocused kernel 2")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel2)
    elif defocus_mm >= 0.5:
        print("Source detect using defocused kernel 3")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel3)
    else:
        print("Source detect using default kernel")
        raw_objects = sep.extract(data, detection_sigma * background_rms, minarea=area_min)

    initial_objects = len(raw_objects)

    raw_objects = Table(raw_objects[np.logical_and.reduce([
        raw_objects['npix'] < area_max,
        # Filter targets near the edge of the frame
        raw_objects['xmin'] > trim_border,
        raw_objects['xmax'] < data.shape[1] - trim_border,
        raw_objects['ymin'] > trim_border,
        raw_objects['ymax'] < data.shape[0] - trim_border
    ])])

    print(detection_sigma * background_rms, initial_objects, len(raw_objects))

    # Astrometry.net expects 1-index pixel positions
    objects = Table()
    objects['X'] = raw_objects['x'] + 1
    objects['Y'] = raw_objects['y'] + 1
    objects['FLUX'] = raw_objects['cflux']
    objects.sort('FLUX')
    objects.reverse()
    return objects


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
    return round(data[int(y - radius):int(y + radius),
                 int(x - radius):int(x + radius)].ravel().max(), 2)


def wcs_phot(data, x, y, rsi, rso, aperture_radii, gain=1.12):
    """
    Take a corrected image array and extract photometry for a set of WCS driven
    X and Y pixel positions. Do this for a series of aperture radii and apply
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
        max_pixel_value = np.array([find_max_pixel_value(data, int(i), int(j), int(r + 1)) for i, j in zip(x, y)])
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
    print(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    print(f"Number of files: {len(filenames)}")

    # Iterate over each filename to get the prefix
    prefixes = get_prefix(filenames)
    print(f"The prefixes are: {prefixes}")

    # Get filenames corresponding to each prefix
    prefix_filenames = [[filename for filename in filenames if filename.startswith(prefix)] for prefix in prefixes]

    for prefix, filenames in zip(prefixes, prefix_filenames):
        print(f"Processing prefix: {prefix}")

        # Check headers for CTYPE1 and CTYPE2
        check_headers(directory, filenames)

        # Calibrate images and get FITS files
        reduced_data, jd_list, bjd_list, hjd_list, prefix_filenames = reduce_images(base_path, out_path)

        # Load WCS information from the first processed image
        first_processed_image = prefix_filenames[0]
        ref_frame_data, ref_header = load_fits_image(first_processed_image)
        print("Header keys:", ref_header.keys())

        ref_frame_bg = sep.Background(ref_frame_data)
        ref_frame_data_corr_no_bg = ref_frame_data - ref_frame_bg
        estimate_coord = SkyCoord(ra=ref_header['TELRA'],
                                  dec=ref_header['TELDEC'],
                                  unit=(u.deg, u.deg))
        estimate_coord_radius = 3 * u.deg

        ref_objects = _detect_objects_sep(ref_frame_data_corr_no_bg, ref_frame_bg.globalrms,
                                          AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
        if len(ref_objects) < N_OBJECTS_LIMIT:
            print(f"Fewer than {N_OBJECTS_LIMIT} found in reference, quitting!")
            sys.exit(TOO_FEW_OBJECTS)

        # Load the photometry catalog
        phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
        print(f"Found catalog with name {prefix}_catalog_input.fits")
        # Convert RA and DEC to pixel coordinates using the WCS information from the header
        wcs = WCS(ref_header)
        phot_x, phot_y = wcs.all_world2pix(phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1)

        print(f"X and Y coordinates: {phot_x}, {phot_y}")


if __name__ == "__main__":
    main()
