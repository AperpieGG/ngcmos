"""
Functions to process CMOS data and chip coordinates
Copyrights 2024 James McCormac, Ioannis Apergis All Rights Reserved

This file contains functions to process CMOS
data and chip coordinates for the NGTS project.
"""
import os
import sys
import math
from datetime import datetime, timedelta
import sep
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import EarthLocation
from skyfield.api import Topos

# pylint: disable = invalid-name
# pylint: disable = redefined-outer-name
# pylint: disable = no-member
# pylint: disable = too-many-locals
# pylint: disable = too-many-arguments
# pylint: disable = unused-variable
# pylint: disable = line-too-long
# pylint: disable = logging-fstring-interpolation


def get_location():
    """
    Get the location of the observatory

    Parameters
    ----------
    None

    Returns
    -------
    site_location : EarthLocation
        location of the observatory

    Raises
    ------
    None
    """
    site_location = EarthLocation(
        lat=-24.615662 * u.deg,
        lon=-70.391809 * u.deg,
        height=2433 * u.m)

    site_topos = Topos(
        latitude_degrees=site_location.lat.to(u.deg).value,
        longitude_degrees=site_location.lon.to(u.deg).value,
        elevation_m=site_location.height.to(u.m).value)

    return site_location, site_topos


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    if not then use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str or None
        Path to the current night directory if found, otherwise None.
    """

    # Get the previous date directory in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(directory, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        # Use the current working directory
        return os.getcwd()


def load_apertures():
    """
    Load the apertures from the config file

    Parameters
    ----------
    None

    Returns
    -------
    apertures : array-like
        array of apertures

    Raises
    ------
    None
    """
    apertures = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    return apertures

