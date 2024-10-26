#!/usr/bin/env python

import batman
import numpy as np
import matplotlib.pyplot as plt
import json

from utils import plot_images, bin_time_flux_error

plot_images()
# Set up the transit parameters
params = batman.TransitParams()
params.t0 = 2455443.06132  # time of inferior conjunction (BJD)
params.per = 4.1567758  # orbital period (days)
params.rp = 0.069  # planet radius (in units of stellar radii)
params.a = 8.84  # semi-major axis (in units of stellar radii)
params.inc = 89.57  # orbital inclination (degrees)
params.ecc = 0.  # eccentricity
params.w = 0  # longitude of periastron (degrees)
params.u = [0.3695, 0.2774]  # limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"  # limb darkening model

# load data from json file
with open('target_light_curve_9725627_CCD.json', 'r') as json_file:
    data = json.load(json_file)

tic_id = data['TIC_ID']
time = np.array(data['Time_BJD'])
flux = np.array(data['Relative_Flux'])
flux_err = np.array(data['Relative_Flux_err'])

# Normalize the flux to the out-of-transit baseline
out_of_transit_mask = (time < (params.t0 - 0.2)) | (time > (params.t0 + 0.2))
baseline_flux = np.median(flux[out_of_transit_mask])
normalized_flux = flux / baseline_flux

# Continue with binning the time and flux
time_binned, flux_binned, fluxerr_binned = bin_time_flux_error(time, normalized_flux, flux_err, 30)

# Normalize the time array to be centered around the transit
time_centered = time_binned - params.t0  # Ensure you use binned time for modeling

# Initialize the transit model with the centered time array
m = batman.TransitModel(params, time_centered)
model_flux = m.light_curve(params) + baseline_flux  # Add baseline flux to the model


# Plot both the observed flux and the model flux to compare
plt.figure()
# plt.errorbar(time, flux, yerr=flux_err, fmt='.', label="Unbinned", color="grey", alpha=0.5)
plt.plot(time, flux, '.', label="Unbinned", color="grey", alpha=0.5)
# plt.errorbar(time_binned, flux_binned, yerr=fluxerr_binned, fmt='o', label="5 min bin", color="red")
plt.plot(time_binned, flux_binned, 'o', label="5 min bin", color="red")
plt.plot(time_binned, model_flux, label="Transit Model", color="black", linestyle='-')
plt.xlabel("Time (BJD)")
plt.ylabel("Relative flux")
plt.legend()
plt.show()
