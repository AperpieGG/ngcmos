#!/usr/bin/env python
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_images

plot_images()


def load_json(file_path):
    """
    Load JSON file and return data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def main():
    # Load the JSON data
    cmos_file = 'flux_vs_temperature_CMOS.json'
    ccd_file = 'flux_vs_temperature_CCD.json'

    print(f"Loading CMOS data from {cmos_file}...")
    cmos_data = load_json(cmos_file)
    print(f"Loading CCD data from {ccd_file}...")
    ccd_data = load_json(ccd_file)

    # Create dictionaries keyed by TIC_ID for quick lookups
    cmos_dict = {entry["TIC_ID"]: entry for entry in cmos_data}
    ccd_dict = {entry["TIC_ID"]: entry for entry in ccd_data}

    # Match TIC_IDs and calculate the flux ratio (CMOS/CCD)
    temperatures = []
    flux_ratios = []
    tmags = []

    for tic_id, cmos_entry in cmos_dict.items():
        if tic_id in ccd_dict:
            ccd_entry = ccd_dict[tic_id]

            # Calculate the flux ratio
            cmos_flux = cmos_entry["Converted_Flux"]
            ccd_flux = ccd_entry["Converted_Flux"]
            flux_ratio = cmos_flux / ccd_flux

            # Append the data
            temperatures.append(cmos_entry["Teff"])
            flux_ratios.append(flux_ratio)
            tmags.append(cmos_entry["Tmag"])  # Use Tmag from CMOS (assuming consistent)

    # Convert to numpy arrays for easier handling
    temperatures = np.array(temperatures)
    flux_ratios = np.array(flux_ratios)
    tmags = np.array(tmags)

    # Plotting
    print("Creating the plot...")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        temperatures, flux_ratios, c=tmags, cmap='cividis', edgecolor='k', alpha=0.75
    )
    plt.colorbar(scatter, label='Tmag (Apparent Magnitude)')
    plt.xlabel('Teff (K)')
    plt.ylabel('CMOS/CCD Flux Ratio')
    plt.title('CMOS/CCD Flux Ratio as a Function of Temperature')
    plt.ylim(0.8, 1.5)
    plt.xlim(3000, 7500)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()