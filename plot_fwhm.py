#! /usr/bin/env python
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_images

plot_images()


# Function to load and process data from JSON file
def load_and_normalize_fwhm(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    bjds = [entry["BJD"] for entry in data["results"]]
    airmass = [entry["Airmass"] for entry in data["results"]]
    fwhm = [entry["FWHM"] for entry in data["results"]]
    median_fwhm = np.median(fwhm)
    normalized_fwhm = [f / median_fwhm for f in fwhm]
    return bjds, airmass, normalized_fwhm


# Load and process data from both JSON files
bjds1, airmass1, fwhm1 = load_and_normalize_fwhm('fwhm_CMOS.json')
bjds2, airmass2, fwhm2 = load_and_normalize_fwhm('fwhm_CCD.json')

# Plot FWHM vs BJD with airmass on secondary x-axis
fig, ax1 = plt.subplots()

# Plot for the first file
ax1.plot(bjds1, fwhm1, 'o', label=f'FWHM CMOS', color='red', alpha=0.5)

# Plot for the second file
ax1.plot(bjds2, fwhm2, 'o', label=f'FWHM CCD', color='blue', alpha=0.5)

# Set main x-axis and y-axis labels
ax1.set_xlabel("BJD")
ax1.set_ylabel("Normalized FWHM")

# Airmass on top x-axis
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Airmass')

# Interpolate airmass values based on ticks on the main x-axis
# We use the airmass from the first file as they are the same for both
interpolated_airmass = np.interp(ax1.get_xticks(), bjds1, airmass1)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')

# Show legend and plot
ax1.legend()
plt.tight_layout()
plt.show()
