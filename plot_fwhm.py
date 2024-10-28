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


# Calculate FWHM ratio
fwhm_ratio = np.array(fwhm1) / np.array(fwhm2)

# Plotting FWHM vs BJD with Airmass as secondary x-axis, and FWHM ratio
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Plot FWHM for File 1 and File 2
ax1.plot(bjds1, fwhm1, 'o', label=f'FWHM CMOS', color='red')
ax1.plot(bjds2, fwhm2, 'o', label=f'FWHM CCD', color='blue')

# Set labels for main plot
ax1.set_ylabel("Normalized FWHM")
ax1.legend()

# Secondary x-axis for Airmass
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Airmass')
interpolated_airmass = np.interp(ax1.get_xticks(), bjds1, airmass1)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')

# Plot the FWHM ratio
ax3.plot(bjds1, fwhm_ratio, 'o', color='green', label='FWHM Ratio (CMOS / CCD)')
ax3.set_xlabel("BJD")
ax3.set_ylabel("FWHM Ratio")
ax3.legend()
# Show the plot
plt.tight_layout()
plt.show()