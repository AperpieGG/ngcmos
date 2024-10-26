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


# Normalize the time array to be centered around the transit
time_centered = time - params.t0

# Initialize the transit model with the centered time array
m = batman.TransitModel(params, time)
model_flux = m.light_curve(params)

# Plot both the observed flux and the model flux to compare
plt.figure(figsize=(10, 6))
plt.plot(time, flux, '.', label="Unbinned Flux", color="grey", alpha=0.5)
plt.plot(time, model_flux, label="Transit Model", color="black", linestyle='-')
plt.xlabel("Time (BJD)")
plt.ylabel("Relative flux")
plt.legend()
plt.show()