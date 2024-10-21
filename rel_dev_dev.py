#!/usr/bin/env python
import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from utils import plot_images, read_phot_file, bin_time_flux_error  # Assuming read_phot_file is available in utils

# Constants for filtering stars
COLOR_TOLERANCE = 0.2
MAGNITUDE_TOLERANCE = 2

plot_images()


def target_info(table, tic_id_to_plot, APERTURE):
    target_star = table[table['tic_id'] == tic_id_to_plot]  # Extract the target star data
    target_tmag = target_star['Tmag'][0]  # Extract the TESS magnitude of the target star
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]  # Extract the color index
    airmass_list = target_star['airmass']  # Extract airmass_list from target star

    # Calculate mean flux for the target star (specific to the chosen aperture)
    target_flux_mean = target_star[f'flux_{APERTURE}'].mean()

    return target_tmag, target_color_index, airmass_list, target_flux_mean


def limits_for_comps(table, tic_id_to_plot, APERTURE):
    # Get target star info including the mean flux
    target_tmag, target_color, airmass_list, target_flux_mean = target_info(table, tic_id_to_plot, APERTURE)

    # Filter based on color index within the tolerance
    color_index = table['gaiabp'] - table['gaiarp']
    color_mask = np.abs(color_index - target_color) <= COLOR_TOLERANCE
    color_data = table[color_mask]

    # Further filter based on TESS magnitude within the tolerance
    mag_mask = np.abs(color_data['Tmag'] - target_tmag) <= MAGNITUDE_TOLERANCE
    valid_color_mag_table = color_data[mag_mask]

    # Exclude stars with Tmag less than 9.4 and remove the target star from the table
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

        if len(mag_nodes) == 0 or len(std_medians) == 0:
            print("No valid standard medians found.")
            break

        # Fit a spline to the medians
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


def find_best_comps(table, tic_id_to_plot, APERTURE):
    # Filter the table based on color/magnitude tolerance
    filtered_table, airmass = limits_for_comps(table, tic_id_to_plot, APERTURE)
    tic_ids = np.unique(filtered_table['tic_id'])
    print(f'Number of comparison stars after the filter table in terms of color/mag: {len(tic_ids)}')

    comp_fluxes = []
    comp_mags = []

    for tic_id in tic_ids:
        flux = filtered_table[filtered_table['tic_id'] == tic_id][f'flux_{APERTURE}']
        tmag = filtered_table[filtered_table['tic_id'] == tic_id]['Tmag'][0]

        comp_fluxes.append(flux)
        comp_mags.append(tmag)

    # Convert lists to arrays for further processing
    comp_fluxes = np.array(comp_fluxes)
    comp_mags = np.array(comp_mags)

    # Check if comp_mags is non-empty before proceeding
    if len(comp_mags) == 0:
        raise ValueError("No valid comparison stars found after filtering for flux and magnitude.")

    # Call the function to find bad comparison stars
    print(f'The dimensions of these two are: {comp_mags.shape}, {comp_fluxes.shape}')
    comp_star_mask, comp_star_rms, iterations = find_bad_comp_stars(comp_fluxes, airmass, comp_mags)

    # Filter the table based on the mask
    print(f'Star with the min rms: {np.min(comp_star_rms)} and tic_id: {tic_ids[np.argmin(comp_star_rms)]}')

    # Filter tic_ids based on the mask
    good_tic_ids = tic_ids[comp_star_mask]

    # Now filter the table based on these tic_ids
    good_comp_star_table = filtered_table[np.isin(filtered_table['tic_id'], good_tic_ids)]

    # print the rms of the stars that kept there
    print(f"RMS of comparison stars after filtering: {comp_star_rms[comp_star_mask]}")

    print(f"Number of iterations to converge: {iterations}")
    return good_comp_star_table  # Return the filtered table including only good comp stars


def plot_comp_lc(time_list, flux_list, fluxerr_list, tic_ids, batch_size=9):
    """
    Plot the light curves for comparison stars in batches of `batch_size` (9 per figure by default).
    """
    total_stars = len(tic_ids)
    num_batches = int(np.ceil(total_stars / batch_size))  # Calculate how many batches we need

    for batch_num in range(num_batches):
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Create 3x3 grid of subplots
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i in range(batch_size):
            idx = batch_num * batch_size + i
            if idx >= total_stars:
                break  # Exit if we exceed the number of stars to plot

            tic_id = tic_ids[idx]
            ax = axes[i // 3, i % 3]  # Select the correct subplot

            # Get the current comparison star flux and time
            comp_fluxes = flux_list[idx]
            comp_fluxerrs = fluxerr_list[idx]
            comp_time = time_list[idx]

            # Calculate the sum of all fluxes except the current star's flux
            reference_fluxes_comp = np.sum(np.delete(flux_list, i, axis=0), axis=0)
            reference_fluxerrs = np.sqrt(np.sum(fluxerr_list ** 2, axis=0))

            # Normalize the current star's flux by the sum of the other comparison stars' fluxes
            comp_fluxes_dt = comp_fluxes / reference_fluxes_comp
            comp_fluxerrs_dt = np.sqrt(comp_fluxerrs ** 2 + reference_fluxerrs ** 2)

            # Bin the data (optional, can be skipped if not needed)
            comp_time_dt, comp_fluxes_dt_binned, comp_fluxerrs_dt_binned = (
                bin_time_flux_error(comp_time, comp_fluxes_dt, comp_fluxerrs_dt, 12))

            # Plot the light curve in the current subplot
            ax.plot(comp_time_dt, comp_fluxes_dt_binned, 'o', color='blue', alpha=0.8)
            ax.set_title(f'Comparison star: {tic_id}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')

        plt.tight_layout()
        plt.show()


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
    # Add parse for tic_id_to_plot
    parser = argparse.ArgumentParser(description='Plot light curves for a given TIC ID.')
    parser.add_argument('tic_id', type=int, help='TIC ID to plot the light curve for.')
    parser.add_argument('--aper', type=int, default=5, help='Aperture number to use for photometry.')
    # Add argument to provide a txt file if comparison stars are known
    parser.add_argument('--comp_stars', type=str, help='Text file with known comparison stars.')

    args = parser.parse_args()
    tic_id_to_plot = args.tic_id
    APERTURE = args.aper
    current_night_directory = os.getcwd()  # Change this if necessary

    # Read the photometry file
    phot_file = get_phot_files(current_night_directory)
    print(f'Photometry file: {phot_file}')

    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

    if args.comp_stars:
        # Read the file with known comparison stars
        comp_stars_file = args.comp_stars
        comp_stars = np.loadtxt(comp_stars_file, dtype=int)
        # Use the tic_ids directly from the phot_table
        tic_ids = np.intersect1d(comp_stars, np.unique(phot_table['tic_id']))
        print(f'Comparison stars from file: {tic_ids}')
    else:
        # Find the best comparison stars
        best_comps_table = find_best_comps(phot_table, tic_id_to_plot, APERTURE)
        tic_ids = np.unique(best_comps_table['tic_id'])

    time_list = []
    flux_list = []
    fluxerr_list = []

    # Collect time, flux, and flux error data
    for tic_id in tic_ids:
        # If comparison stars were loaded from the file, ensure best_comps_table is created
        if args.comp_stars:
            best_comps_table = find_best_comps(phot_table, tic_id_to_plot, APERTURE)

        comp_time = best_comps_table[best_comps_table['tic_id'] == tic_id]['jd_mid']
        comp_fluxes = best_comps_table[best_comps_table['tic_id'] == tic_id][f'flux_{APERTURE}']
        comp_fluxerrs = best_comps_table[best_comps_table['tic_id'] == tic_id][f'fluxerr_{APERTURE}']

        time_list.append(comp_time)
        flux_list.append(comp_fluxes)
        fluxerr_list.append(comp_fluxerrs)

    # Convert lists to arrays
    flux_list = np.array(flux_list)
    fluxerr_list = np.array(fluxerr_list)
    time_list = np.array(time_list)

    # Reference fluxes and errors (sum of all stars, excluding the target star)
    reference_fluxes = np.sum(flux_list, axis=0)
    reference_fluxerrs = np.sqrt(np.sum(fluxerr_list ** 2, axis=0))

    # Bin the master reference data
    time_list_binned, reference_fluxes_binned, reference_fluxerrs_binned = (
        bin_time_flux_error(time_list[0], reference_fluxes, reference_fluxerrs, 12))

    # Call the plot function
    plot_comp_lc(time_list, flux_list, fluxerr_list, tic_ids)

    # Perform relative photometry for target star and plot
    target_star = phot_table[phot_table['tic_id'] == tic_id_to_plot]
    target_flux = target_star[f'flux_{APERTURE}']
    target_fluxerr = target_star[f'fluxerr_{APERTURE}']
    target_time = target_star['jd_mid']

    # Bin the target star data
    target_time_binned, target_fluxes_binned, target_fluxerrs_binned = (
        bin_time_flux_error(target_time, target_flux, target_fluxerr, 12))

    # Calculate the flux ratio for the target star with respect to the summation of the reference stars' fluxes
    flux_ratio_binned = target_fluxes_binned / reference_fluxes_binned
    flux_ratio = target_flux / reference_fluxes
    # Calculate the average flux ratio of the target star
    flux_ratio_mean_binned = np.mean(flux_ratio_binned)
    flux_ratio_mean = np.mean(flux_ratio)
    # Normalize the flux ratio (result around unity)
    target_fluxes_dt = flux_ratio_binned / flux_ratio_mean_binned
    target_fluxes_dt_unbinned = flux_ratio / flux_ratio_mean
    RMS = np.std(target_fluxes_dt_unbinned)
    RMS_binned = np.std(target_fluxes_dt)
    print(f'Target star has an RMS of {RMS:.4f} before binning and {RMS_binned:.4f} after binning.')

    plt.plot(target_time_binned, target_fluxes_dt, 'o', color='red', label=f'RMS unbinned = {RMS:.4f}')
    plt.title(f'Target star: {tic_id_to_plot} divided by master')
    plt.legend(loc='best')
    plt.show()

    if APERTURE == 5:
        camera = 'CMOS'
    elif APERTURE == 4:
        camera = 'CCD'

    # Save target_time_binned and target_fluxes_dt in a JSON file
    data_to_save = {
        "time": target_time_binned.tolist(),
        "flux": target_fluxes_dt.tolist()
    }

    json_filename = f'target_light_curve_{tic_id_to_plot}_{camera}.json'
    with open(json_filename, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    print(f'Data saved to {json_filename}')


# Run the main function
if __name__ == "__main__":
    main()