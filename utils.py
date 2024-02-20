"""
Functions for handling on-sky or on chip coordinates
"""
import sys
import math
from datetime import datetime
from astropy.io import fits
import sep
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
import astropy.units as u
import jastro.housekeeping as jhk


# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=c-extension-no-member


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


def catalogue_to_pixels(filenames, ra_dec_coords):
    """
    Convert a list of catalogue positions to X and Y image
    coordinates

    Parameters
    ----------
    filenames : list of filenames
    ra_dec_coords : list of (RA, Dec) coordinate tuples

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Arrays of X and Y pixel coordinates
    """
    try:
        _, hdr = fits.getdata(filenames, header=True)
    except (OSError, IndexError, KeyError) as e:
        print(f'CANNOT FIND HEADER INFORMATION IN {filenames}, EXITING...')
        sys.exit(1)

    # Load the WCS
    w = WCS(hdr)

    # Convert RA and Dec to pixel coordinates
    pix = w.wcs_world2pix(ra_dec_coords, 0)
    x, y = pix[:, 0], pix[:, 1]

    return x, y


def parse_region_content(region_content):
    """
    Parse the content of a region file to extract RA and Dec coordinates.

    Parameters
    ----------
    region_content : str
        Content of the region file.

    Returns
    -------
    list of tuples
        List of (RA, Dec) coordinate tuples extracted from the region content.
    """
    ra_dec_coords = []
    # Split the region content into lines
    lines = region_content.split('\n')
    for line in lines:
        # Assuming each line represents a circular region with RA and Dec coordinates
        # Extract the RA and Dec coordinates from the line
        if line.startswith('circle'):
            # Example line format: "circle(123.456, 78.901, 2.0)"
            parts = line.split('(')[1].split(')')[0].split(',')
            ra = float(parts[0])
            dec = float(parts[1])
            ra_dec_coords.append((ra, dec))
    return ra_dec_coords


# TODO: Fix the following functions for the pipeline
def extract_background(filename, sigma):
    """
    Extract background from the image using SEP

    Parameters
    ----------
    filename : str
        The name of the file to run sep.extract on
    sigma : int, optional
        The number of sigma a detection must be above the background
        to be flagged as a star

    Returns
    -------
    bkg : array-like
        Background values extracted from the image

    Raises
    ------
    None
    """
    data, _ = jhk.load_fits_image(filename)
    data = data.astype(np.float64)

    bkg = sep.Background(data)
    return bkg


def source_extract(filename, sigma, ap_radius, catalogue_coords):
    """
    Measure the sky background and perform photometry on sources
    in the image supplied

    Parameters
    ----------
    filename : str
        The name of the file to run sep.extract on
    sigma : int, optional
        The number of sigma a detection must be above the background
        to be flagged as a star
    ap_radius : float
        Radius of the circular aperture for photometry
    catalogue_coords : array-like
        RA and Dec in degrees of the targets positions

    Returns
    -------
    results : list
        List containing dictionaries with photometry results

    Raises
    ------
    None
    """
    data, hdr = jhk.load_fits_image(filename)
    data = data.astype(np.float64)

    bkg = extract_background(filename, sigma)
    data_sub_bkg = data - bkg

    # perform photometry on sources
    phot_results = []
    for coord in catalogue_coords:
        x, y = catalogue_to_pixels(filename, [coord])
        result = sep.sum_circle(data_sub_bkg, x[0], y[0], ap_radius)
        phot_results.append(result)

    return phot_results


def compute_background_flux(filename, ap_radius, catalogue_coords, sigma):
    """
    Compute background flux using circular apertures

    Parameters
    ----------
    filename : str
        The name of the FITS file
    ap_radius : float
        Radius of the circular aperture for photometry
    catalogue_coords : array-like
        RA and Dec in degrees of the background regions

    Returns
    -------
    background_flux : float
        Background flux value

    Raises
    ------
    None
    """
    data, hdr = jhk.load_fits_image(filename)
    data = data.astype(np.float64)

    bkg = extract_background(filename, sigma)
    data_sub_bkg = data - bkg

    background_flux = []
    for coord in catalogue_coords:
        x, y = catalogue_to_pixels(filename, [coord])
        result = sep.sum_circle(data_sub_bkg, x[0], y[0], ap_radius)
        background_flux.append(result)

    return background_flux


def output_results_to_file(photometry_results, background_flux, filename, ap_radius):
    """
    Output photometry results and background flux to a text file

    Parameters
    ----------
    photometry_results : list
        List containing dictionaries with photometry results
    background_flux : float
        Background flux value
    filename : str
        Name of the output text file

    Returns
    -------
    None

    Raises
    ------
    None
    """
    with open(filename, 'w') as f:
        f.write("Object_ID Flux Flux_Error Background Background_Signal\n")
        for idx, result in enumerate(photometry_results):
            flux = result['flux']
            fluxerr = result['fluxerr']
            bkg = background_flux[idx]
            bkg_signal = bkg * ap_radius ** 2 * np.pi
            f.write(f"{idx + 1} {flux} {fluxerr} {bkg} {bkg_signal}\n")
