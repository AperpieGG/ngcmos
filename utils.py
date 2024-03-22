"""
Functions for handling on-sky or on chip coordinates
"""
import fnmatch
import os
from datetime import datetime, timedelta
from astropy.io import fits
import sep
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
import astropy.units as u
from astropy.table import Table, hstack
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation


# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=c-extension-no-member


def plot_images():
    # Set plot parameters
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    # Font and fontsize
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14


def get_location():
    """
    Get the location of the observatory

    Parameters
    ----------
    None

    Returns
    -------
    loc : EarthLocation
        Location of the observatory

    Raises
    ------
    None
    """
    site_location = EarthLocation(
        lat=-24.615662 * u.deg,
        lon=-70.391809 * u.deg,
        height=2433 * u.m
    )

    return site_location


def get_light_travel_times(ra, dec, time_to_correct):
    """
    Get the light travel times to the helio- and
    barycentric

    Parameters
    ----------
    ra : str
        The Right Ascension of the target in hour-angle
        e.g. 16:00:00
    dec : str
        The Declination of the target in degrees
        e.g. +20:00:00
    time_to_correct : astropy.Time object
        The time of observation to correct. The astropy.Time
        object must have been initialised with an EarthLocation

    Returns
    -------
    ltt_bary : float
        The light travel time to the barycentre
    ltt_helio : float
        The light travel time to the heliocentric

    Raises
    ------
    None
    """
    target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ltt_bary = time_to_correct.light_travel_time(target)
    ltt_helio = time_to_correct.light_travel_time(target, 'heliocentric')
    return ltt_bary, ltt_helio


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
    data, header = fits.getdata(filename, header=True, ext=ext)
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
    if 0.15 <= defocus_mm < 0.3:
        print("Source detect using defocused kernel 1")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel1)
    elif 0.3 <= defocus_mm < 0.5:
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


def get_phot_files(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'phot_*.fits'):
            phot_files.append(filename)
    return phot_files


def read_phot_file(filename):
    """
    Read the photometry file.

    Parameters
    ----------
    filename : str
        Photometry file to read.

    Returns
    -------
    astropy.table.table.Table
        Table containing the photometry data.
    """
    # Read the photometry file here using fits or any other appropriate method
    try:
        with fits.open(filename) as ff:
            # Access the data in the photometry file as needed
            tab = ff[1].data
            return tab
    except Exception as e:
        print(f"Error reading photometry file {filename}: {e}")
        return None


def bin_time_flux_error(time, flux, error, bin_fact):
    """
    Use reshape to bin light curve data, clip under filled bins
    Works with 2D arrays of flux and errors

    Note: under filled bins are clipped off the end of the series

    Parameters
    ----------
    time : array         of times to bin
    flux : array         of flux values to bin
    error : array         of error values to bin
    bin_fact : int
        Number of measurements to combine

    Returns
    -------
    times_b : array
        Binned times
    flux_b : array
        Binned fluxes
    error_b : array
        Binned errors

    Raises
    ------
    None
    """
    n_binned = int(len(time) / bin_fact)
    clip = n_binned * bin_fact
    time_b = np.average(time[:clip].reshape(n_binned, bin_fact), axis=1)
    # determine if 1 or 2d flux/err inputs
    if len(flux.shape) == 1:
        flux_b = np.average(flux[:clip].reshape(n_binned, bin_fact), axis=1)
        error_b = np.sqrt(np.sum(error[:clip].reshape(n_binned, bin_fact) ** 2, axis=1)) / bin_fact
    else:
        # assumed 2d with 1 row per star
        n_stars = len(flux)
        flux_b = np.average(flux[:clip].reshape((n_stars, n_binned, bin_fact)), axis=2)
        error_b = np.sqrt(np.sum(error[:clip].reshape((n_stars, n_binned, bin_fact)) ** 2, axis=2)) / bin_fact
    return time_b, flux_b, error_b


def remove_outliers(time, flux, flux_err):
    """
    Remove massive outliers in 3 rounds of clipping
    """
    n_time = np.copy(time)
    n_flux = np.copy(flux)
    n_flux_err = np.copy(flux_err)

    for _ in range(3):
        mad = median_abs_deviation(n_flux)
        loc = np.where(((n_flux < np.median(n_flux) + 10 * mad) & (n_flux > np.median(n_flux) - 10 * mad)))[0]

        # Update time, flux, and flux_err with non-outlier values
        n_time = n_time[loc]
        n_flux = n_flux[loc]
        n_flux_err = n_flux_err[loc]

        # If no outliers were removed in this round, return the filtered arrays
        if len(n_time) == len(time):
            return n_time, n_flux, n_flux_err

    return n_time, n_flux, n_flux_err





