#! /usr/bin/env python
import os
from datetime import datetime, timedelta
import numpy as np
from calibration_images import reduce_images
from utils import (get_location, _detect_objects_sep, get_catalog,
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
import argparse

parser = argparse.ArgumentParser(description="Run photometry for a specific TIC ID.")
parser.add_argument("--tic", type=int, required=True, help="Target TIC ID")
args = parser.parse_args()

TARGET_TIC = args.tic


GAIN = 1.131
MAX_ALLOWED_PIXEL_SHIFT = 50
N_OBJECTS_LIMIT = 200
APERTURE_RADII = [5]
RSI = 15  # 15
RSO = 20  # 20
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


def wcs_phot(data, x, y, aperture_radii, frame_data_corr_no_bg, bg_rms, gain):
    """
    Extract photometry at positions (x, y) for multiple aperture radii.
    Optionally uses a precomputed mask to prevent contamination in the annulus background.

    Parameters
    ----------
    data : 2D array
        Image data.
    x, y : array-like
        Pixel coordinates of sources.
    bg_rms : float
        Background RMS for error estimation.
    frame_data_corr_no_bg : 2D array
        Precomputed mask to prevent contamination in the annulus background.
    aperture_radii : array
        Aperture radii for photometry.
    gain : float
        CCD/CMOS gain.
    Returns
    -------
    Tout : astropy Table
        Photometry table with fluxes, errors, max pixel in aperture.
    """

    # Column labels
    col_labels = ["flux", "fluxerr", "flux_w_sky", "fluxerr_w_sky", "max_pixel_value"]
    Tout = None

    for r in aperture_radii:
        # Sum flux inside aperture with background annulus
        flux, fluxerr, _ = sep.sum_circle(frame_data_corr_no_bg, x, y, r,
                                          subpix=0,
                                          err=bg_rms,
                                          gain=gain)
        flux_w_sky, fluxerr_w_sky, _ = sep.sum_circle(data, x, y, r,
                                                      subpix=0,
                                                      gain=gain)
        # Maximum pixel value inside the aperture
        max_pixel_value = np.array([data[int(yi), int(xi)] for xi, yi in zip(x, y)])

        # Build the table
        row_data = [flux, fluxerr, flux_w_sky, fluxerr_w_sky, max_pixel_value]
        if Tout is None:
            Tout = Table(row_data,
                         names=[f"{c}_{r}" for c in col_labels])
        else:
            T = Table(row_data,
                      names=[f"{c}_{r}" for c in col_labels])
            Tout = hstack([Tout, T])

    return Tout


def wcs_phot_annulus(data, x, y, rsi, rso, aperture_radii, gain):
    """
    Extract photometry at positions (x, y) for multiple aperture radii.
    Optionally uses a precomputed mask to prevent contamination in the annulus background.

    Parameters
    ----------
    data : 2D array
        Image data.
    x, y : array-like
        Pixel coordinates of sources.
    rsi, rso : float
        Inner and outer radius of background annulus.
    aperture_radii : list of float
        Aperture radii for photometry.
    gain : float
        CCD/CMOS gain.
    Returns
    -------
    Tout : astropy Table
        Photometry table with fluxes, errors, max pixel in aperture.
    """

    # Column labels
    col_labels = ["flux", "fluxerr", "flux_w_sky", "fluxerr_w_sky", "max_pixel_value"]
    Tout = None

    for r in aperture_radii:
        # Sum flux inside aperture with background annulus
        flux, fluxerr, _ = sep.sum_circle(data, x, y, r,
                                          subpix=0,
                                          bkgann=(rsi, rso),
                                          gain=gain)
        flux_w_sky, fluxerr_w_sky, _ = sep.sum_circle(data, x, y, r,
                                                      subpix=0,
                                                      gain=gain)
        # Maximum pixel value inside the aperture
        max_pixel_value = np.array([data[int(yi), int(xi)] for xi, yi in zip(x, y)])

        # Build the table
        row_data = [flux, fluxerr, flux_w_sky, fluxerr_w_sky, max_pixel_value]
        if Tout is None:
            Tout = Table(row_data,
                         names=[f"{c}_{r}" for c in col_labels])
        else:
            T = Table(row_data,
                      names=[f"{c}_{r}" for c in col_labels])
            Tout = hstack([Tout, T])

    return Tout


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


# set directory for the current working directory
directory = os.getcwd()
logging.info(f"Directory: {directory}")

# filter filenames only for .fits data files
filenames = filter_filenames(directory)
logging.info(f"Number of files: {len(filenames)}")


# Get prefixes for each set of images
for prefix in get_prefix(filenames):
    for filename in filenames:
        print(f"Processing filename {filename}......")
        # Calibrate image and get FITS file
        print(f"The average pixel value for {filename} is {fits.getdata(os.path.join(directory, filename)).mean()}")
        reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])
        print(f"The average pixel value for {filename} is {reduced_data[0].mean()}")
        # Convert reduced_data to a dictionary with filenames as keys
        reduced_data_dict = {filename: (data, header) for data, header in zip(reduced_data, reduced_header)}

        # Access the reduced data and header corresponding to the filename
        frame_data, frame_hdr = reduced_data_dict[filename]
        print(f"Extracting photometry for {filename}")

        # Extract airmass and zero point from the header
        airmass, zp = extract_airmass_and_zp(frame_hdr)

        frame_bg = sep.Background(frame_data)
        # calculate background rms
        bg_rms = frame_bg.rms()
        frame_data_corr_no_bg = frame_data - frame_bg
        frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms,
                                            AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
        if len(frame_objects) < N_OBJECTS_LIMIT:
            logging.info(f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!")
            continue

        # Load the photometry catalog
        phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)

        # Select only target TIC
        mask = phot_cat['tic_id'] == TARGET_TIC

        if not np.any(mask):
            print(f"TIC {TARGET_TIC} not found in catalog.")
            continue

        phot_cat = phot_cat[mask]

        print(f"Found target TIC {TARGET_TIC} in catalog.")

        logging.info(f"Found catalog with name {prefix}_catalog_input.fits")
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

        frame_preamble = Table([frame_ids, phot_cat['Tmag'], phot_cat['tic_id'],
                                phot_cat['gaiabp'], phot_cat['gaiarp'], time_jd.value, time_bary.value,
                                phot_x, phot_y,
                                [airmass] * len(phot_x), [zp] * len(phot_x)],
                               names=("frame_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                                      "jd_bary", "x", "y", "airmass", "zp"))

        # Extract photometry at locations
        # Background model subtraction photometry
        phot_bg = wcs_phot(
            frame_data,
            phot_x,
            phot_y,
            APERTURE_RADII,
            frame_data_corr_no_bg,
            bg_rms,
            gain=GAIN
        )

        # Annulus sky photometry
        phot_ann = wcs_phot_annulus(
            frame_data,
            phot_x,
            phot_y,
            RSI,
            RSO,
            APERTURE_RADII,
            gain=GAIN
        )

        for r in APERTURE_RADII:
            # Background-subtracted
            flux_bg = phot_bg[f"flux_{r}"][0]
            flux_w_sky_bg = phot_bg[f"flux_w_sky_{r}"][0]
            sky_bg = flux_w_sky_bg - flux_bg

            # Annulus
            flux_ann = phot_ann[f"flux_{r}"][0]
            flux_w_sky_ann = phot_ann[f"flux_w_sky_{r}"][0]
            sky_ann = flux_w_sky_ann - flux_ann

            print(f"\n===== TIC {TARGET_TIC} | Aperture {r} =====")
            print("Background Model Method:")
            print(f"Flux (no sky): {flux_bg:.2f}")
            print(f"Sky contribution: {sky_bg:.2f}")
            print(f"Total contribution (flux + sky): {flux_w_sky_bg:.2f}")

            print("\nAnnulus Method:")
            print(f"Flux (no sky): {flux_ann:.2f}")
            print(f"Sky contribution: {sky_ann:.2f}")
            print(f"Total contribution (flux + sky): {flux_w_sky_ann:.2f}")