#!/usr/bin/env python

import batman
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils import plot_images, bin_time_flux_error

plot_images()
# Set up the transit parameters
params = batman.TransitParams()
params.t0 = 2458326.10418  # time of inferior conjunction (BJD)
params.per = 0.9809734  # orbital period (days)
params.rp = 0.287  # planet radius (in units of stellar radii)
params.a = 5.289  # semi-major axis (in units of stellar radii)
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
flux = tic_id_data['Relative_Flux']
flux_err = tic_id_data['Relative_Flux_err']

time_binned, flux_binned, fluxerr_binned = bin_time_flux_error(time, flux, flux_err, 30)
# Normalize the time array to be centered around the transit
time_centered = time - params.t0

# Initialize the transit model with the centered time array
m = batman.TransitModel(params, time_binned)
model_flux = m.light_curve(params)

# Plot both the observed flux and the model flux to compare
plt.figure(figsize=(10, 6))
plt.plot(time, flux, '.', label="Observed Flux", color="grey", alpha=0.5)
plt.plot(time_binned, flux_binned, 'o', label="Bin 5 min", color="yellow")
plt.plot(time_binned, model_flux, label="Transit Model", color="black", linestyle='-')
plt.xlabel("Time (days) from central transit")
plt.ylabel("Relative flux")
plt.legend()
plt.show()