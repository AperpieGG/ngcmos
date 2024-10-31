#!/usr/bin/env python

import batman
import numpy as np
import matplotlib.pyplot as plt
import json

from utils import plot_images, bin_time_flux_error
import argparse

parser = argparse.ArgumentParser(description='Plot the transit model for a given TIC ID and camera number.')
parser.add_argument('cam', type=str, help='Camera number (CCD or CMOS)')
parser.add_argument('target', type=str, help='Target name')
args = parser.parse_args()
cam = args.cam
target = args.target

plot_images()
# Set up the transit parameters
params = batman.TransitParams()
params.t0 = 2456338.44251  # time of inferior conjunction (BJD)
params.per = 2.1846730  # orbital period (days)
params.rp = 0.1076  # planet radius (in units of stellar radii)
params.a = 6.47  # semi-major axis (in units of stellar radii)
params.inc = 88.4  # orbital inclination (degrees)
params.ecc = 0.  # eccentricity
params.w = 0  # longitude of periastron (degrees)
params.u = [0.4412, 0.2312]  # limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"  # limb darkening model

# load data from json file
with open(f'target_light_curve_{target}_{cam}.json', 'r') as json_file:
    data = json.load(json_file)

tic_id = data['TIC_ID']
time = np.array(data['Time_BJD'])
flux = np.array(data['Relative_Flux'])
flux_err = np.array(data['Relative_Flux_err'])

time_binned, flux_binned, fluxerr_binned = bin_time_flux_error(time, flux, flux_err, 60)
# Calculate the mean of the normalized data
mean_dt_flux = np.mean(flux_binned)

# Calculate the adjustment needed to bring it to a baseline of 1
# adjustment = mean_dt_flux - 0.998 # CCD
adjustment = mean_dt_flux - 0.997  # CMOS


# Adjust the normalized flux data
dt_flux_adjusted = flux_binned - adjustment

# Normalize the time array to be centered around the transit
time_centered = time - params.t0

# Initialize the transit model with the centered time array
m = batman.TransitModel(params, time_binned)
model_flux = m.light_curve(params)

# Plot both the observed flux and the model flux to compare
plt.figure()
# plt.errorbar(time, flux, yerr=flux_err, fmt='.', label="Unbinned", color="grey", alpha=0.5)
plt.plot(time, flux, '.', label="Unbinned", color="grey", alpha=0.5)
# plt.errorbar(time_binned, flux_binned, yerr=fluxerr_binned, fmt='o', label="5 min bin", color="red")
plt.plot(time_binned, dt_flux_adjusted, 'o', label="5 min bin", color="red")
plt.plot(time_binned, model_flux, label="Transit Model", color="black", linestyle='-')
plt.xlabel("Time (BJD)")
plt.ylabel("Relative flux")
plt.legend()
plt.show()
