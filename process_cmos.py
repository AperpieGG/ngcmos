#! /usr/bin/env python
import os
from collections import defaultdict
from datetime import datetime, timedelta
from calibration_images import reduce_images
from donuts import Donuts
import numpy as np
from utils import get_location, wcs_phot, _detect_objects_sep, get_catalog
import json
import warnings
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.wcs import WCS
import sep
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

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
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot"]
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


def check_donuts(file_groups, filenames):
    """
    Check donuts for each group of images with the same prefix.

    Parameters
    ----------
    file_groups : list of str
        Prefixes for the groups of images.
    filenames : list of str
        List of lists of filenames for the groups of images.
    """
    for filename, file_group in zip(filenames, file_groups):
        # Using the first filename as the reference image
        reference_image = file_group[0]
        print(f"Reference image: {reference_image}")

        # Assuming Donuts class and measure_shift function are defined elsewhere
        d = Donuts(reference_image)

        for filename in file_group[1:]:
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
        # Check headers for CTYPE1 and CTYPE2
        check_headers(directory, filenames)

        # Check donuts for each group
        check_donuts(prefix_filenames, filenames)

        # Calibrate images and get FITS files
        reduced_data, reduced_header, prefix_filenames = reduce_images(base_path, out_path)

        # Convert reduced_data to a dictionary with filenames as keys
        reduced_data_dict = {filename: (data, header) for filename, data, header in
                             zip(filenames, reduced_data, reduced_header)}

        all_photometry = None

        # Iterate over each filename for the current prefix
        for filename in filenames:
            # Access the reduced data and header corresponding to the filename
            frame_data, frame_hdr = reduced_data_dict[filename]
            print(f"Extracting photometry for {filename}")

            wcs_ignore_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'IMAGEW', 'IMAGEH']
            wcs_header = {}
            for line in [frame_hdr[i:i + 80] for i in range(0, len(frame_hdr), 80)]:
                key = line[0:8].strip()
                if '=' in line and key not in wcs_ignore_cards:
                    card = fits.Card.fromstring(line)
                    wcs_header[card.keyword] = card.value

            frame_bg = sep.Background(frame_data)
            frame_data_corr_no_bg = frame_data - frame_bg
            estimate_coord = SkyCoord(ra=frame_hdr['TELRA'],
                                      dec=frame_hdr['TELDEC'],
                                      unit=(u.deg, u.deg))
            estimate_coord_radius = 3 * u.deg

            frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms,
                                                AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
            if len(frame_objects) < N_OBJECTS_LIMIT:
                print(f"Fewer than {N_OBJECTS_LIMIT} found in {filename}, skipping!")
                continue

            # Load the photometry catalog
            phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
            print(f"Found catalog with name {prefix}_catalog.fits")
            # Convert RA and DEC to pixel coordinates using the WCS information from the header

            phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1)

            print(f"X and Y coordinates: {phot_x}, {phot_y}")

            # Do time conversions - one time value per format per target
            half_exptime = frame_hdr['EXPTIME'] / 2.
            time_isot = Time([frame_hdr['DATE-OBS'] for i in range(len(phot_x))],
                             format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            # Correct to mid-exposure time
            time_jd = time_jd + half_exptime * u.second
            ra = phot_cat['ra_deg_corr']
            dec = phot_cat['dec_deg_corr']

            frame_ids = [filename for i in range(len(phot_x))]
            print(f"Found {len(frame_ids)} frames")

            frame_preamble = Table([frame_ids, phot_cat['gaia_id'], time_jd.value, phot_x, phot_y],
                                   names=("frame_id", "gaia_id", "jd_mid", "x", "y"))

            # Extract photometry at locations
            frame_phot = wcs_phot(frame_data, phot_x, phot_y, RSI, RSO, APERTURE_RADII, gain=GAIN)

            # Stack the photometry and preamble
            frame_output = hstack([frame_preamble, frame_phot])

            # Define the filename for the photometry output
            phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")  # Include directory path

            # Convert frame_output to a Table if it's not already
            if not isinstance(frame_output, Table):
                frame_output = Table(frame_output)

            # Append the current frame's photometry to the accumulated photometry
            if all_photometry is None:
                all_photometry = frame_output
            else:
                all_photometry = vstack([all_photometry, frame_output])

        # Save the photometry
        if all_photometry is not None:
            all_photometry.write(phot_output_filename, overwrite=True)
            print(f"Saved photometry to {phot_output_filename}")
        else:
            print("No photometry data to save.")

        print("Done!")


if __name__ == "__main__":
    main()

