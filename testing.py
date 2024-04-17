#!/usr/bin/env python
from astropy.io import fits
import matplotlib.pyplot as plt

# Load the FITS file containing the relative photometry data
fits_filename = 'rel_phot_NG0625+0058.fits'  # Replace 'XXXX-XXXX' with the appropriate filename
data_table = fits.getdata(fits_filename)

# Extract RMS and Tmag values from the data table
rms_values = data_table['RMS']
tmag_values = data_table['Tmag']

# Plot RMS versus Tmag
plt.figure(figsize=(8, 6))
plt.scatter(tmag_values, rms_values, color='blue', alpha=0.5)
plt.xlabel('Tmag')
plt.ylabel('RMS')
plt.yscale('log')
plt.title('RMS vs Tmag')
plt.show()