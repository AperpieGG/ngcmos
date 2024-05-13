#! /usr/bin/env python
import os
from datetime import datetime, timedelta
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


# ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)


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
        prefix = filename[:11]
        prefixes.add(prefix)
    return prefixes


def main():
    # set directory for the current working directory
    directory = os.getcwd()
    print(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    print(f"Number of files: {len(filenames)}")

    # Get prefixes for each set of images
    prefixes = get_prefix(filenames)
    print(f"The prefixes are: {prefixes}")

    for prefix in prefixes:
        phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

        # Open the photometry file for the current prefix
        if os.path.exists(phot_output_filename):
            print(f"Photometry file for prefix {prefix} already exists, skipping to the next prefix.\n")
            continue

        print(f"Creating new photometry file for prefix {prefix}.\n")
        phot_table = None

        # Iterate over filenames with the current prefix
        prefix_filenames = [filename for filename in filenames if filename.startswith(prefix)]
        for filename in prefix_filenames:
            print(f"Processing filename {filename}......")
            # Calibrate image and get FITS file
            print(f"The average pixel value for {filename} is {fits.getdata(os.path.join(directory, filename)).mean()}")
            reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])
            print(f"The average pixel value for {filename} is {reduced_data[0].mean()}")
            # Convert reduced_data to a dictionary with filenames as keys
            reduced_data_dict = {filename: (data, header) for data, header in zip(reduced_data, reduced_header)}

            # Access the reduced data and header corresponding to the filename
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
            estimate_coord = SkyCoord(ra=frame_hdr['TELRA'],
                                      dec=frame_hdr['TELDEC'],
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
                                   names=("frame_id", "gaia_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid", "x", "y"))

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
