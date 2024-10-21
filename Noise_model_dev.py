#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils import plot_images

plot_images()

def get_phot_files(directory):
    """
    Function to retrieve photometry files from a given directory.
    Returns a list of filenames.
    """
    files = [f for f in os.listdir(directory) if f.startswith('phot') and f.endswith('.fits')]
    if len(files) == 0:
        raise FileNotFoundError("No FITS files found in the directory.")
    return files  # Return the list of FITS files found


def read_phot_files(filename):
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
    try:
        with fits.open(filename) as ff:
            tab = ff[1].data
            return tab
    except Exception as e:
        print(f"Error reading photometry file {filename}: {e}")
        return None


def find_stars(table, APERTURE):
    """
    Find stars in the photometry table and calculate RMS and Tmag for each star.

    Parameters
    ----------
    table : astropy.table.Table
        Table containing photometry data.
    APERTURE : str
        The aperture to use for flux data.

    Returns
    -------
    comp_star_rms : array
        RMS values for stars.
    selected_Tmags : array
        Tmag values for stars.
    selected_tic_ids : array
        TIC IDs for stars.
    """
    unique_tic_ids = np.unique(table['tic_id'])  # Get unique tic_id values
    airmass = table['airmass']

    comp_star_rms = []
    selected_Tmags = []
    selected_tic_ids = []

    for tic_id in unique_tic_ids:
        tic_mask = [tic_id == i for i in table['tic_id'] if i < 14]
        fluxes = table[f'flux_{APERTURE}'][tic_mask]
        Tmag = table['Tmag'][tic_mask][0]

        # Fit airmass and calculate RMS
        airmass_cs = np.polyfit(airmass[tic_mask], fluxes, 1)
        airmass_mod = np.polyval(airmass_cs, airmass[tic_mask])
        flux_corrected = fluxes / airmass_mod
        flux_norm = flux_corrected / np.median(flux_corrected)
        rms_val = np.std(flux_norm)

        comp_star_rms.append(rms_val)
        selected_Tmags.append(Tmag)
        selected_tic_ids.append(tic_id)

    return np.array(comp_star_rms), np.array(selected_Tmags), np.array(selected_tic_ids)


def plot_rms_vs_tmag(cmos_rms, cmos_tmags, ccd_rms=None, ccd_tmags=None):
    """
    Plot RMS vs Tmag for both CMOS and CCD data.

    Parameters
    ----------
    cmos_rms : np.array
        RMS values for CMOS stars.
    cmos_tmags : np.array
        Tmag values for CMOS stars.
    ccd_rms : np.array, optional
        RMS values for CCD stars.
    ccd_tmags : np.array, optional
        Tmag values for CCD stars.
    """
    plt.figure(figsize=(10, 6))

    # Plot CMOS
    plt.scatter(cmos_tmags, cmos_rms, color='red', label='CMOS', alpha=0.7)

    if ccd_rms is not None and ccd_tmags is not None:
        # Plot CCD
        plt.scatter(ccd_tmags, ccd_rms, color='blue', label='CCD', alpha=0.7)

    # Add labels and title
    plt.xlabel('Tmag')
    plt.ylabel('RMS')
    plt.gca().invert_xaxis()  # Invert Tmag axis (brighter stars have lower Tmag)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to process CMOS and then CCD photometry files, calculate RMS and Tmag,
    and plot the results sequentially to save time.
    """

    # Get the list of photometry files (assumes one is CMOS and one is CCD)
    directory = os.getcwd()
    phot_files = get_phot_files(directory)

    # Find the CMOS and CCD files (this assumes there's only one of each)
    cmos_file = [f for f in phot_files if 'CMOS' in f][0]
    ccd_file = [f for f in phot_files if 'CCD' in f][0]

    # Read and process the CMOS file to find stars, RMS, and Tmag
    print("Processing CMOS data...")
    cmos_table = read_phot_files(os.path.join(directory, cmos_file))
    cmos_rms, cmos_tmags, cmos_tic_ids = find_stars(cmos_table, APERTURE=5)

    # Plot the CMOS data first
    print("Plotting CMOS data...")
    plot_rms_vs_tmag(cmos_rms, cmos_tmags)

    # Now, read the CCD data after having the CMOS tic_ids
    print("Processing CCD data...")
    ccd_table = read_phot_files(os.path.join(directory, ccd_file))
    ccd_mask = np.isin(ccd_table['tic_id'], cmos_tic_ids)
    ccd_rms, ccd_tmags, _ = find_stars(ccd_table[ccd_mask], APERTURE=4)

    # Plot the combined CMOS and CCD data
    print("Plotting combined CMOS and CCD data...")
    plot_rms_vs_tmag(cmos_rms, cmos_tmags, ccd_rms, ccd_tmags)


if __name__ == "__main__":
    main()