#! /usr/bin/env python
import logging
import os
import numpy as np
from utils import (get_location, wcs_phot, _detect_objects_sep, get_catalog,
                   extract_airmass_and_zp, get_light_travel_times)
import json
import warnings
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.wcs import WCS
import sep
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
import fitsio


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


GAIN = 2
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
        if filename.endswith('.fits') and not filename.endswith('.fits.bz2'):
            fits_path = os.path.join(directory, filename)
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                if 'IMGCLASS' in header and header['IMGCLASS'] == 'SCIENCE':
                    filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def get_prefix(filenames, directory):
    """
    Extract unique prefixes from a list of filenames based on the OBJECT keyword in FITS headers.

    Parameters
    ----------
    filenames : list of str
        List of filenames.
    directory : str
        Directory containing the FITS files.

    Returns
    -------
    set of str
        Set of unique prefixes extracted from the OBJECT keyword in the FITS headers.
    """
    fits_path = os.path.join(directory, filenames[0])
    with fits.open(fits_path) as hdul:
        prefix = hdul[0].header.get('OBJECT', '')
    return prefix


def load_fits(filename, ext=0):
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
    data, header = fits.getdata(filename, header=True, ext=ext)

    # Ensure data is converted to float if it's not already
    if data.dtype.kind != 'f':
        try:
            data = data.astype(float)
        except ValueError:
            raise ValueError("Unable to convert data to float. Data may contain non-numeric values.")

    return data, header


def main():
    # get the current working directory
    parent_directory = os.getcwd()

    # get a list of subdirectories inside the parent directory
    subdirectories = [name for name in os.listdir(parent_directory) if
                      os.path.isdir(os.path.join(parent_directory, name))]
    subdirectories = [subdirectory for subdirectory in subdirectories if
                      subdirectory.startswith("action") and subdirectory.endswith("_observeField")]

    logging.info(f'The subdirectories are: {subdirectories}')

    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_observeField"):
            # form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # set directory for the current subdirectory
            directory = subdirectory_path
            logging.info(f"Directory: {directory}")

            # filter filenames only for .fits data files
            filenames = filter_filenames(directory)
            logging.info(f"Number of files: {len(filenames)}")

            # Get prefixes for each set of images
            prefix = get_prefix(filenames, directory)
            # prefix for the specific subdirectory
            logging.info(f"The prefix for the {subdirectory} is: {prefix}")

            phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

            if os.path.exists(phot_output_filename):
                logging.info(f"Photometry file for prefix {prefix} already exists, skipping to the next prefix.\n")
                continue

            logging.info(f"Creating new photometry file for prefix {prefix}.\n")
            phot_table = None

            for filename in filenames:
                logging.info(f"Processing filename {filename}......")
                # Calibrate image and get FITS file
                ref_frame_data, ref_header = load_fits(os.path.join(directory, filename))
                logging.info(f"The average pixel value for {filename} is {fits.getdata(os.path.join(directory, filename)).mean()}")

                if os.path.exists(os.path.join(parent_directory, 'master_bias.fits')):
                    master_bias = fits.getdata(os.path.join(parent_directory, 'master_bias.fits'))

                if master_bias is not None:
                    # Check if master_bias has the same dimensions as ref_frame_data
                    if ref_frame_data.shape == master_bias.shape:
                        ref_frame_data_corr = ref_frame_data - master_bias
                        logging.info(
                            f"After bias subtraction, mean pixel value for {filename}: {np.mean(ref_frame_data_corr)}")
                    else:
                        logging.info(
                            f"Master bias shape {master_bias.shape} does not match frame shape {ref_frame_data.shape}!")
                        # Trim ref_frame_data to match the dimensions of master_bias
                        if master_bias.shape == (2048, 2048):
                            ref_frame_data = ref_frame_data[:, 20:2068]
                            logging.info(f"Trimmed ref_frame_data shape to match master_bias "
                                         f"shape: {ref_frame_data.shape}")
                            ref_frame_data_corr = ref_frame_data - master_bias
                            logging.info(
                                f"After bias subtraction, mean pixel value for {filename}: "
                                f"{np.mean(ref_frame_data_corr)}")
                        else:
                            logging.info("Unable to trim ref_frame_data: master_bias shape is not (2048, 2088)")
                            continue
                else:
                    # Handle the case where master_bias is not assigned
                    logging.info("Master bias not found, skipping bias subtraction.")
                    ref_frame_data_corr = ref_frame_data  # No bias subtraction

                # Convert reduced_data to a dictionary with filenames as keys
                reduced_data_dict = {filename: (ref_frame_data_corr, ref_header)}

                frame_data, frame_hdr = reduced_data_dict[filename]
                logging.info(f"Extracting photometry for {filename}\n")

                airmass, zp = extract_airmass_and_zp(frame_hdr)

                wcs_ignore_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'IMAGEW', 'IMAGEH']
                wcs_header = {}
                for line in [frame_hdr[i:i + 80] for i in range(0, len(frame_hdr), 80)]:
                    key = line[0:8].strip()
                    if '=' in line and key not in wcs_ignore_cards:
                        card = fits.Card.fromstring(line)
                        wcs_header[card.keyword] = card.value

                frame_bg = sep.Background(frame_data)
                frame_data_corr_no_bg = frame_data - frame_bg
                estimate_coord = SkyCoord(ra=frame_hdr['CMD_RA'],
                                          dec=frame_hdr['CMD_DEC'],
                                          unit=(u.deg, u.deg))
                estimate_coord_radius = 3 * u.deg

                frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms,
                                                    AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
                if len(frame_objects) < N_OBJECTS_LIMIT:
                    logging.info(f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!\n")
                    continue

                # Load the photometry catalog
                phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
                logging.info(f"Found catalog with name {prefix}_catalog_input.fits\n")
                # Convert RA and DEC to pixel coordinates using the WCS information from the header
                phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1)

                # Do time conversions - one time value per format per target
                half_exptime = frame_hdr['EXPTIME'] / 2.
                time_isot = Time([frame_hdr['DATE-OBS'] for i in range(len(phot_x))],
                                 format='isot', scale='utc', location=get_location())
                time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
                # Correct to mid-exposure time
                time_jd = time_jd + half_exptime * u.second
                ra = phot_cat['ra_deg_corr']
                dec = phot_cat['dec_deg_corr']
                ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
                time_bary = time_jd.tdb + ltt_bary
                time_helio = time_jd.utc + ltt_helio

                frame_ids = [filename for i in range(len(phot_x))]
                logging.info(f"Found {len(frame_ids)} sources")

                frame_preamble = Table([frame_ids, phot_cat['gaia_id'], phot_cat['Tmag'], phot_cat['tic_id'],
                                        phot_cat['gaiabp'], phot_cat['gaiarp'], time_jd.value, time_bary.value,
                                        time_helio.value, phot_x, phot_y,
                                        [airmass] * len(phot_x), [zp] * len(phot_x)],
                                       names=("frame_id", "gaia_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                                              "jd_bary", "jd_helio", "x", "y", "airmass", "zp"))

                # Extract photometry at locations
                frame_phot = wcs_phot(frame_data, phot_x, phot_y, RSI, RSO, APERTURE_RADII, gain=GAIN)

                # Stack the photometry and preamble
                frame_output = hstack([frame_preamble, frame_phot])

                # Convert frame_output to a Table if it's not already
                if not isinstance(frame_output, Table):
                    frame_output = Table(frame_output)

                # Append the current frame's photometry to the accumulated photometry
                if phot_table is None:
                    phot_table = frame_output
                else:
                    phot_table = vstack([phot_table, frame_output])

                logging.info(f"Finished photometry for {filename}\n")

                # Save the photometry for the current prefix
            if phot_table is not None:
                phot_table.write(phot_output_filename, overwrite=True)
                logging.info(f"Saved photometry for prefix {prefix} to {phot_output_filename}\n")
            else:
                logging.info(f"No photometry data for prefix {prefix}.\n")

            logging.info("Done!\n")


if __name__ == "__main__":
    main()
