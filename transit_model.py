#!/usr/bin/env python

import batman
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils import plot_images

plot_images()
# Set up the transit parameters
params = batman.TransitParams()
params.t0 = 2458326.10418  # time of inferior conjunction (BJD)
params.per = 0.9809734  # orbital period (days)
params.rp = 0.204  # planet radius (in units of stellar radii)
params.a = 0.01782  # semi-major axis (in units of stellar radii)
params.inc = 77.18  # orbital inclination (degrees)
params.ecc = 0.  # eccentricity
params.w = 178  # longitude of periastron (degrees)
params.u = [0.545, 0.195]  # limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"  # limb darkening model

# Load the time and flux data from the FITS file
with fits.open('rel_phot_HIP-65-A_1.fits') as hdul:
    table = hdul[1].data

# Filter for the specific TIC ID (if needed)
tic_id_data = table[table['TIC_ID'] == 201248411]
time = tic_id_data['Time_JD']
observed_flux = tic_id_data['Relative_Flux']  # Assuming your FITS data has flux column

# Normalize the time array to be centered around the transit
time_centered = time - params.t0

# Initialize the transit model with the centered time array
m = batman.TransitModel(params, time_centered)
model_flux = m.light_curve(params)

# Plot both the observed flux and the model flux to compare
plt.figure(figsize=(10, 6))
plt.plot(time_centered, observed_flux, 'o', label="Observed Flux", color="blue")
plt.plot(time_centered, model_flux, label="Transit Model", color="red", linestyle='-')
plt.xlabel("Time (days) from central transit")
plt.ylabel("Relative flux")
plt.legend()
plt.show()