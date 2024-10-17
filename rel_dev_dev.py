#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from utils import plot_images, read_phot_file  # Assuming read_phot_file is available in utils

# Constants for filtering stars
SIGMA = 2
COLOR_TOLERANCE = 0.2
MAGNITUDE_TOLERANCE = 0.5
APERTURE = 5


def target_info(table, tic_id_to_plot):
    target_star = table[table['tic_id'] == tic_id_to_plot]  # Extract the target star data
    target_tmag = target_star['Tmag'][0]  # Extract the TESS magnitude of the target star
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]  # Extract the color index
    airmass_list = target_star['airmass']  # Extract airmass_list from target star
    return target_tmag, target_color_index, airmass_list


def limits_for_comps(table, tic_id_to_plot):
    target_tmag, target_color, airmass_list = target_info(table, tic_id_to_plot)
    color_index = table['gaiabp'] - table['gaiarp']
    color_mask = np.abs(color_index - target_color) <= COLOR_TOLERANCE
    color_data = table[color_mask]
    mag_mask = np.abs(color_data['Tmag'] - target_tmag) <= MAGNITUDE_TOLERANCE
    valid_color_mag_table = color_data[mag_mask]
    valid_color_mag_table = valid_color_mag_table[valid_color_mag_table['Tmag'] > 9.4]
    filtered_table = valid_color_mag_table[valid_color_mag_table['tic_id'] != tic_id_to_plot]
    return filtered_table, airmass_list


def find_comp_star_rms(comp_fluxes, airmass):
    comp_star_rms = []
    for i, flux in enumerate(comp_fluxes):
        airmass_cs = np.polyfit(airmass, flux, 1)
        airmass_mod = np.polyval(airmass_cs, airmass)
        flux_corrected = flux / airmass_mod
        flux_norm = flux_corrected / np.median(flux_corrected)
        rms_val = np.std(flux_norm)
        comp_star_rms.append(rms_val)
    return np.array(comp_star_rms)


def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0, sig_level=3., dmag=0.5):
    comp_star_rms = find_comp_star_rms(comp_fluxes, airmass)
    comp_star_mask = np.array([True for cs in comp_star_rms])
    i = 0.
    while True:
        i += 1.
        comp_mags = np.copy(comp_mags0[comp_star_mask])
        comp_rms = np.copy(comp_star_rms[comp_star_mask])
        N1 = len(comp_mags)
        edges = np.arange(comp_mags.min(), comp_mags.max() + dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1] + edges[1:]) / 2.
        std_medians = np.array([np.nan if len(comp_rms[dig == i]) == 0
                                else np.median(comp_rms[dig == i])
                                for i in range(1, len(edges))])
        cut = np.isnan(std_medians)
        mag_nodes = mag_nodes[~cut]
        std_medians = std_medians[~cut]
        spl = Spline(mag_nodes, std_medians)
        mod = spl(comp_mags)
        mod0 = spl(comp_mags0)
        std = np.std(comp_rms - mod)
        comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        N2 = np.sum(comp_star_mask)
        if N1 == N2:
            break
        elif i > 10.:
            break
    return comp_star_mask, comp_star_rms, i


def find_best_comps(table, tic_id_to_plot):
    filtered_table, airmass = limits_for_comps(table, tic_id_to_plot)
    tic_ids = np.unique(filtered_table['tic_id'])
    print(f'Number of comparison stars after the filter table in terms of color/mag: {len(tic_ids)}')

    comp_fluxes = []
    comp_mags = []
    for tic_id in tic_ids:
        flux = filtered_table[filtered_table['tic_id'] == tic_id][f'flux_{APERTURE}']
        zero_point_list = filtered_table[filtered_table['tic_id'] == tic_id]['zp']
        exp = 10
        comp_fluxes.append(flux)
        mag = -2.5 * np.log10(flux / exp) + zero_point_list
        comp_mags.append(mag)

    comp_fluxes = np.array(comp_fluxes)
    comp_mags = np.array(comp_mags)

    # Call the function to find bad comparison stars
    comp_star_mask, comp_star_rms, iterations = find_bad_comp_stars(comp_fluxes, airmass, comp_mags)

    # Filter the table based on the mask
    good_comp_star_table = filtered_table[comp_star_mask]

    print(f"Number of iterations to converge: {iterations}")
    return good_comp_star_table  # Return the filtered table including only good comp stars


def get_phot_files(directory):
    """
    Function to retrieve the first photometry file from a given directory.
    Returns the filename as a string.
    """
    files = [f for f in os.listdir(directory) if f.startswith('phot') and f.endswith('.fits')]
    if len(files) == 0:
        raise FileNotFoundError("No FITS files found in the directory.")
    return files[0]  # Return the first FITS file found as a string


def main():
    # Set the target TIC ID and the current night directory
    tic_id_to_plot = 9725627
    current_night_directory = os.getcwd()  # Change this if necessary

    # Read the photometry file
    phot_file = get_phot_files(current_night_directory)
    print(f'Photometry file: {phot_file}')

    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    print(f'Photometry table: {phot_table}')
    # Find the best comparison stars
    best_comps_table = find_best_comps(phot_table, tic_id_to_plot)

    # Output the best comparison stars
    print("Best comparison stars table:")
    print(best_comps_table)


# Run the main function
if __name__ == "__main__":
    main()
