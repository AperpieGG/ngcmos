#!/usr/bin/env python
"""
Script to plot the RMS vs Tmags for CMOS and CCD. It takes the photometry files
and search for the Tmag and fits a second order polynomial with the airmass,
normalizes the lc and find the RMS for each particular star
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils import plot_images, read_phot_file

plot_images()
# TODO: find best stars for CMOS with the smaller RMS for each mag (use Ed's code for that - find bad comps)
# TODO: add noise source script to also plot the total model for each camera


def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0, sig_level=2., dmag=0.5):
    comp_star_rms = find_comp_star_rms(comp_fluxes, airmass)
    print(comp_star_rms)
    print(f'Number of comparison stars RMS before filtering: {len(comp_star_rms)}')
    comp_star_mask = np.array([True for _ in comp_star_rms])
    i = 0
    while True:
        i += 1
        comp_mags = np.copy(comp_mags0[comp_star_mask])
        comp_rms = np.copy(comp_star_rms[comp_star_mask])
        N1 = len(comp_mags)

        if N1 == 0:
            print("No valid comparison stars left after filtering.")
            break

        edges = np.arange(comp_mags.min(), comp_mags.max() + dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1] + edges[1:]) / 2.

        # Initialize std_medians and populate it based on bins
        std_medians = []
        for j in range(1, len(edges)):
            in_bin = comp_rms[dig == j]
            if len(in_bin) == 0:
                std_medians.append(np.nan)  # No stars in this bin
            else:
                std_medians.append(np.median(in_bin))

        std_medians = np.array(std_medians)

        # Remove NaN entries from std_medians and mag_nodes
        valid_mask = ~np.isnan(std_medians)
        mag_nodes = mag_nodes[valid_mask]
        std_medians = std_medians[valid_mask]

        # Handle case with too few points for spline fitting
        if len(mag_nodes) < 4 or len(std_medians) < 4:  # Less than 4 points
            print("Too few valid points for spline fitting. Falling back to linear fit.")
            if len(mag_nodes) > 1:
                mod = np.interp(comp_mags, mag_nodes, std_medians)  # Use linear interpolation
                mod0 = np.interp(comp_mags0, mag_nodes, std_medians)
            else:
                print("Not enough data for linear interpolation either. Skipping iteration.")
                break
        else:
            # Fit a spline to the medians if enough data
            spl = Spline(mag_nodes, std_medians)
            mod = spl(comp_mags)
            mod0 = spl(comp_mags0)

        std = np.std(comp_rms - mod)
        comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        N2 = np.sum(comp_star_mask)

        # Print the number of stars included and excluded
        print(f"Iteration {i}:")
        print(f"Stars included: {N2}, Stars excluded: {N1 - N2}")

        # Exit condition: no further changes or too many iterations
        if N1 == N2 or i > 10:
            break

    return comp_star_mask, comp_star_rms, i


def get_phot_files(directory):
    """
    Function to retrieve photometry files from a given directory.
    Returns a list of filenames.
    """
    files = [f for f in os.listdir(directory) if f.startswith('phot') and f.endswith('.fits')]
    if len(files) == 0:
        raise FileNotFoundError("No FITS files found in the directory.")
    return files  # Return the list of FITS files found


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
    # Filter for Tmag < 14
    valid_mask = table['Tmag'] < 14
    filtered_table = table[valid_mask]

    unique_tic_ids = np.unique(filtered_table['tic_id'])  # Get unique tic_id values
    airmass = filtered_table['airmass']

    comp_star_rms = []
    selected_Tmags = []
    selected_tic_ids = []

    for tic_id in unique_tic_ids:
        tic_mask = filtered_table['tic_id'] == tic_id
        fluxes = filtered_table[f'flux_{APERTURE}'][tic_mask]
        Tmag = filtered_table['Tmag'][tic_mask][0]

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
    cmos_table = read_phot_file(os.path.join(directory, cmos_file))
    cmos_rms, cmos_tmags, cmos_tic_ids = find_stars(cmos_table, APERTURE=5)

    # Plot the CMOS data first
    print("Plotting CMOS data...")
    plot_rms_vs_tmag(cmos_rms, cmos_tmags)

    # Now, read the CCD data after having the CMOS tic_ids
    print("Processing CCD data...")
    ccd_table = read_phot_file(os.path.join(directory, ccd_file))
    ccd_mask = np.isin(ccd_table['tic_id'], cmos_tic_ids)
    ccd_rms, ccd_tmags, _ = find_stars(ccd_table[ccd_mask], APERTURE=4)

    # Plot the combined CMOS and CCD data
    print("Plotting combined CMOS and CCD data...")
    plot_rms_vs_tmag(cmos_rms, cmos_tmags, ccd_rms, ccd_tmags)


if __name__ == "__main__":
    main()
