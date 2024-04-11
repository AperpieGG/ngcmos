#! /usr/bin/env python
import os

import numpy as np

from calibration_images import reduce_images
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
from astropy.utils.exceptions import AstropyWarning
import fitsio


# ignore some annoying warnings
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

    print('The subdirectories are:', subdirectories)

    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_observeField"):
            # form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # set directory for the current subdirectory
            directory = subdirectory_path
            print(f"Directory: {directory}")

            # filter filenames only for .fits data files
            filenames = filter_filenames(directory)
            print(f"Number of files: {len(filenames)}")

            # Get prefixes for each set of images
            prefix = get_prefix(filenames, directory)
            # print prefix for the specific subdirectory
            print(f"The prefix for the {subdirectory} is: {prefix}")

            phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

            if os.path.exists(phot_output_filename):
                print(f"Photometry file for prefix {prefix} already exists, skipping to the next prefix.\n")
                continue

            print(f"Creating new photometry file for prefix {prefix}.\n")
            phot_table = None

            for filename in filenames:
                print(f"Processing filename {filename}......")
                # Calibrate image and get FITS file
                ref_frame_data, ref_header = load_fits(os.path.join(directory, filename))
                print(f"The average pixel value for {filename} is {fits.getdata(os.path.join(directory, filename)).mean()}")

                # Reduce the image
                print(ref_frame_data.shape)
                if ref_frame_data.shape == (2048, 2088):
                    ref_oscan = np.median(ref_frame_data[:, 2075:], axis=1)
                    ref_frame_data = ref_frame_data[:, 20:2068]
                    ref_frame_data_corr = ref_frame_data - ref_oscan
                else:
                    ref_frame_data_corr = ref_frame_data

                print(f"After overscan subtraction, mean pixel value for {filename}: {np.mean(ref_frame_data_corr)}")
                print(f"The shape of the frame is: {ref_frame_data_corr.shape}")
                # Convert reduced_data to a dictionary with filenames as keys
                reduced_data_dict = {filename: (ref_frame_data_corr, ref_header)}

                frame_data, frame_hdr = reduced_data_dict[filename]
                print(f"Extracting photometry for {filename}\n")

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
                    print(f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!\n")
                    continue

                # Load the photometry catalog
                phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
                print(f"Found catalog with name {prefix}_catalog_input.fits\n")
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

                frame_ids = [filename for i in range(len(phot_x))]
                print(f"Found {len(frame_ids)} sources")

                frame_preamble = Table([frame_ids, phot_cat['gaia_id'], phot_cat['Tmag'], phot_cat['tic_id'],
                                        phot_cat['gaiabp'], phot_cat['gaiarp'], time_jd.value, phot_x, phot_y],
                                       names=(
                                       "frame_id", "gaia_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid", "x", "y"))

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

                print(f"Finished photometry for {filename}\n")

                # Save the photometry for the current prefix
            if phot_table is not None:
                phot_table.write(phot_output_filename, overwrite=True)
                print(f"Saved photometry for prefix {prefix} to {phot_output_filename}\n")
            else:
                print(f"No photometry data for prefix {prefix}.\n")

            print("Done!\n")


if __name__ == "__main__":
    main()
