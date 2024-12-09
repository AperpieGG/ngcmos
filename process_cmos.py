#! /usr/bin/env python
import os
from datetime import datetime, timedelta
from calibration_images import reduce_images
from utils import (get_location, wcs_phot, _detect_objects_sep, get_catalog,
                   extract_airmass_and_zp, get_light_travel_times)
import json
import warnings
import logging
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.wcs import WCS
import sep
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

# Set up logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the overall logging level

# Create file handler
file_handler = logging.FileHandler('process.log')
file_handler.setLevel(logging.INFO)  # Set the level for the file handler

# Create stream handler (for terminal output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Set the level for the stream handler

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

GAIN = 1.131
MAX_ALLOWED_PIXEL_SHIFT = 50
N_OBJECTS_LIMIT = 200
APERTURE_RADII = [4, 4.9, 5, 6, 8, 10]
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
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot", "catalog_input"]
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
        with fits.open(filename) as hdulist:
            object_keyword = hdulist[0].header.get('OBJECT', '')
            prefix = object_keyword[:11]  # Take first 11 letters
            if prefix:  # Check if prefix is not empty
                prefixes.add(prefix)
    return prefixes


def main():
    # set directory for the current working directory
    directory = os.getcwd()
    logging.info(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    logging.info(f"Number of files: {len(filenames)}")

    # Get prefixes for each set of images
    prefixes = get_prefix(filenames)
    logging.info(f"The prefixes are: {prefixes}")

    for prefix in prefixes:
        phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

        # Open the photometry file for the current prefix
        if os.path.exists(phot_output_filename):
            logging.info(f"Photometry file for prefix {prefix} already exists, skipping to the next prefix.")
            continue

        logging.info(f"Creating new photometry file for prefix {prefix}.")
        phot_table = None

        # Iterate over filenames with the current prefix
        prefix_filenames = [filename for filename in filenames if filename.startswith(prefix)]
        for filename in prefix_filenames:
            logging.info(f"Processing filename {filename}...")

            try:
                # Calibrate image and get FITS file
                reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])

                frame_data, frame_hdr = reduced_data[0], reduced_header[0]

                # Extract airmass and zero point from the header
                airmass, zp = extract_airmass_and_zp(frame_hdr)

                # Perform background subtraction
                frame_bg = sep.Background(frame_data)
                frame_data_corr_no_bg = frame_data - frame_bg

                # Detect objects
                frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms,
                                                    AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
                if len(frame_objects) < N_OBJECTS_LIMIT:
                    logging.info(f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!")
                    continue

                # Load the photometry catalog
                phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)

                # Convert RA and DEC to pixel coordinates using WCS
                phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1)

                # Perform time conversions
                half_exptime = frame_hdr['EXPTIME'] / 2.0
                time_isot = Time([frame_hdr['DATE-OBS']] * len(phot_x), format='isot', scale='utc',
                                 location=get_location())
                time_jd = time_isot.jd + half_exptime / 86400.0
                ra, dec = phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr']
                ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
                time_bary = time_jd + ltt_bary.jd
                time_helio = time_jd + ltt_helio.jd

                frame_ids = [filename] * len(phot_x)

                # Prepare photometry table
                frame_preamble = Table([
                    frame_ids, phot_cat['gaia_id'], phot_cat['Tmag'], phot_cat['tic_id'],
                    phot_cat['gaiabp'], phot_cat['gaiarp'], time_jd, time_bary, time_helio, phot_x, phot_y,
                    [airmass] * len(phot_x), [zp] * len(phot_x)
                ], names=("frame_id", "gaia_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                          "jd_bary", "jd_helio", "x", "y", "airmass", "zp"))

                # Perform photometry
                frame_phot = wcs_phot(frame_data, phot_x, phot_y, RSI, RSO, APERTURE_RADII, gain=GAIN)

                # Combine preamble and photometry
                frame_output = hstack([frame_preamble, frame_phot])

                # Save or append to the photometry file
                if os.path.exists(phot_output_filename):
                    existing_table = Table.read(phot_output_filename)
                    updated_table = vstack([existing_table, frame_output])
                    updated_table.write(phot_output_filename, overwrite=True)
                else:
                    frame_output.write(phot_output_filename, overwrite=True)

                logging.info(f"Saved photometry for {filename} to {phot_output_filename}")

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

    logging.info("All processing completed.")

if __name__ == "__main__":
    main()
