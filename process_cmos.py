#! /usr/bin/env python
import os
import signal
import sys
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
from multiprocessing import Queue, Process

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


def get_processed_frame_ids_fits(phot_output_filename):
    """Extract processed frame IDs using Astropy FITS."""
    if not os.path.exists(phot_output_filename):
        return set()
    try:
        with fits.open(phot_output_filename) as hdul:
            data = hdul[1].data  # Assuming the table is in the first extension
            return set(data['frame_id'])
    except Exception as e:
        logging.error(f"Error reading {phot_output_filename}: {e}")
        return set()


def file_already_processed_fits(filename, processed_frame_ids):
    """Check if a file has already been processed using preloaded IDs."""
    return filename in processed_frame_ids


# Signal handling for termination
def handle_termination(signum, frame):
    logging.info("Termination signal received. Exiting gracefully.")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_termination)
signal.signal(signal.SIGINT, handle_termination)


def photometry_writer(queue, phot_output_filename):
    """Writes photometry data asynchronously."""
    while True:
        frame_output = queue.get()

        # Check if the processing is done
        if frame_output == "DONE":
            logging.info("Writer process completed.")
            break

        # Try saving the photometry data
        try:
            if isinstance(frame_output, Table):
                if os.path.exists(phot_output_filename):
                    existing_table = Table.read(phot_output_filename)
                    updated_table = vstack([existing_table, frame_output])
                else:
                    updated_table = frame_output

                updated_table.write(phot_output_filename, overwrite=True)
                logging.info(f"Saved photometry to {phot_output_filename}")
            else:
                logging.error("Received invalid frame output, skipping save.")

        except Exception as e:
            logging.error(f"Error saving photometry data: {e}")


# Improved save function
def save_photometry_incremental(phot_output_filename, frame_output):
    try:
        if os.path.exists(phot_output_filename):
            frame_output.write(phot_output_filename, format='fits', append=True)
        else:
            frame_output.write(phot_output_filename, overwrite=True)
        logging.info(f"Saved photometry to {phot_output_filename}")
    except Exception as e:
        logging.error(f"Error saving photometry: {e}")


# Main processing function
def process_file(queue, directory, prefix, filename, phot_output_filename, processed_frame_ids):
    try:
        if file_already_processed_fits(filename, processed_frame_ids):
            logging.info(f"File {filename} already processed, skipping...")
            return

        logging.info(f"Processing {filename}...")

        # Perform photometry on the file
        reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])
        reduced_data_dict = {filename: (data, header) for data, header in zip(reduced_data, reduced_header)}

        # Access the reduced data and header
        frame_data, frame_hdr = reduced_data_dict[filename]

        # Extract airmass and zero point from the header
        airmass, zp = extract_airmass_and_zp(frame_hdr)

        # Perform WCS calculations
        phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
        phot_x, phot_y = WCS(frame_hdr).all_world2pix(
            phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1
        )

        frame_ids = [filename] * len(phot_x)
        # Perform time conversions
        half_exptime = frame_hdr['EXPTIME'] / 2.
        time_isot = Time(
            [frame_hdr['DATE-OBS'] for _ in range(len(phot_x))],
            format='isot', scale='utc', location=get_location()
        )
        time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
        time_jd = time_jd + half_exptime * u.second

        ra = phot_cat['ra_deg_corr']
        dec = phot_cat['dec_deg_corr']
        ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
        time_bary = time_jd.tdb + ltt_bary
        time_helio = time_jd.utc + ltt_helio

        frame_ids = [filename] * len(phot_x)
        logging.info(f"Found {len(frame_ids)} sources")

        frame_preamble = Table(
            [frame_ids, phot_cat['gaia_id'], phot_cat['Tmag'], phot_cat['tic_id'],
             phot_cat['gaiabp'], phot_cat['gaiarp'], time_jd.value, time_bary.value,
             time_helio.value, phot_x, phot_y,
             [airmass] * len(phot_x), [zp] * len(phot_x)],
            names=("frame_id", "gaia_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                   "jd_bary", "jd_helio", "x", "y", "airmass", "zp")
        )

        frame_phot = wcs_phot(frame_data, phot_x, phot_y, RSI, RSO, APERTURE_RADII, gain=GAIN)
        frame_output = hstack([frame_preamble, frame_phot])

        # Save results asynchronously
        queue.put(frame_output)
        logging.info(f"Finished photometry for {filename}")

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")


def main():
    directory = os.getcwd()
    logging.info(f"Directory: {directory}")

    filenames = filter_filenames(directory)
    logging.info(f"Number of files: {len(filenames)}")

    prefixes = get_prefix(filenames)
    logging.info(f"The prefixes are: {prefixes}")

    for prefix in prefixes:
        phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

        processed_frame_ids = get_processed_frame_ids_fits(phot_output_filename)
        logging.info(f"Loaded {len(processed_frame_ids)} processed frames for prefix {prefix}.")

        prefix_filenames = [filename for filename in filenames if filename.startswith(prefix)]

        queue = Queue()
        writer_process = Process(target=photometry_writer, args=(queue, phot_output_filename))
        writer_process.start()

        for filename in prefix_filenames:
            process_file(queue, directory, prefix, filename, phot_output_filename, processed_frame_ids)

        queue.put("DONE")
        writer_process.join()

    logging.info("Photometry process completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Photometry process interrupted by user.")