#!/usr/bin/env python
import argparse
import datetime
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from utils import (plot_images, find_current_night_directory, get_phot_files, read_phot_file,
                   bin_time_flux_error, remove_outliers)


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


def linear_model(x, m, b):
    return m * x + b


def calculate_mean_rms_flux(table, num_stars):
    mean_flux_list = []
    tmag_list = []

    for tic_id in table['tic_id'][:num_stars]:  # Selecting the first num_stars stars
        tic_id_data = table[table['tic_id'] == tic_id]
        Tmag = tic_id_data['Tmag'][0]
        jd_mid = tic_id_data['jd_mid']
        flux_6 = tic_id_data['flux_6']
        fluxerr_6 = tic_id_data['fluxerr_6']

        time_clipped, flux_6_clipped, fluxerr_6_clipped = remove_outliers(jd_mid, flux_6, fluxerr_6)
        mean_flux = np.mean(flux_6_clipped)

        if mean_flux > 0:  # Filter out zero or negative flux values
            mean_flux_list.append(mean_flux)
            tmag_list.append(Tmag)

    mean_flux = np.array(mean_flux_list)
    tmag = np.array(tmag_list)

    # Perform curve fitting
    popt, pcov = curve_fit(linear_model, tmag, np.log(mean_flux))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(tmag, mean_flux, 'o', color='black', alpha=0.5)
    plt.plot(tmag, np.exp(linear_model(tmag, *popt)), '--', color='red', label='Linear fit')
    plt.gca().invert_xaxis()
    plt.xlabel('Tmag')
    plt.ylabel('Mean Flux (log scale)')
    plt.title('Tmag vs Mean Flux')
    plt.yscale('log')
    plt.legend()

    # Extract slope and intercept from optimized parameters
    slope, intercept = popt

    print(f"Slope (zeropoint) and intercept: {slope}, {intercept}")

    plt.show()

    return slope, intercept, mean_flux, tmag


def save_results_to_json(results, phot_file):
    slope, intercept, mean_flux, tmag = results

    # Create data dictionary
    data = {
        "slope": slope,
        "intercept": intercept,
        "mean_flux": mean_flux.tolist(),
        "tmag": tmag.tolist()
    }

    # Save data to JSON file
    filename = f'zeropoint_{phot_file}.json'
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)

    print(f"Results saved to {filename}")


def main(phot_file):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('--num_stars', type=int, default=100, help='Number of stars to plot')
    args = parser.parse_args()

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Plot the current photometry file
    print(f"Plotting the photometry file {phot_file}...")
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

    results = calculate_mean_rms_flux(phot_table, num_stars=args.num_stars)

    # Save results to JSON file
    save_results_to_json(results, phot_file)


def main_loop(phot_files):
    for phot_file in phot_files:
        main(phot_file)


if __name__ == "__main__":
    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Run the main function for each photometry file
    main_loop(phot_files)
