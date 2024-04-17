#!/usr/bin/env python
from astropy.io import fits
import matplotlib.pyplot as plt
from utils import plot_images

plot_images()
# Load the FITS file containing the relative photometry data
fits_filename = 'rel_phot_NG1109-2807.fits'  # Replace 'XXXX-XXXX' with the appropriate filename
data_table = fits.getdata(fits_filename)

# Extract time and flux for the first star
TIC_ID = data_table['TIC_ID'][2]  # TIC ID for the first star
first_star_time = data_table['Time_JD'][2]  # Time for the first star
first_star_flux = data_table['Relative_Flux'][2]  # Flux for the first star

# Plot flux versus time for the first star
plt.figure(figsize=(8, 6))
plt.plot(first_star_time, first_star_flux, 'o')
plt.xlabel('Time (JD)')
plt.ylabel('Relative Flux')
plt.title('Relative Photometry for TIC ID ' + str(TIC_ID))
plt.show()