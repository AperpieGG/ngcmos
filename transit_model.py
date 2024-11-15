#!/usr/bin/env python

import batman
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import plot_images, bin_time_flux_error
import argparse

# Argument parser for input
parser = argparse.ArgumentParser(description='Plot the transit model for a given TIC ID and camera numbers.')
parser.add_argument('cam1', type=str, help='First camera number (CCD or CMOS)')
parser.add_argument('cam2', type=str, help='Second camera number (CCD or CMOS)')
parser.add_argument('target', type=str, help='Target name')
args = parser.parse_args()
cam1 = args.cam1
cam2 = args.cam2
target = args.target

plot_images()


# Function to process data and generate plots
def process_camera(cam, target):
    # Set up the transit parameters
    params = batman.TransitParams()
    params.t0 = 2456338.44251  # time of inferior conjunction (BJD)
    params.per = 2.1846730  # orbital period (days)
    params.rp = 0.1024  # planet radius (in units of stellar radii)
    params.a = 6.47  # semi-major axis (in units of stellar radii)
    params.inc = 88.4  # orbital inclination (degrees)
    params.ecc = 0  # eccentricity
    params.w = 0  # longitude of periastron (degrees)
    params.u = [0.4412, 0.2312]  # limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"  # limb darkening model

    # Load data from JSON file
    with open(f'target_light_curve_{target}_{cam}.json', 'r') as json_file:
        data = json.load(json_file)

    tic_id = data['TIC_ID']
    time = np.array(data['Time_BJD'])
    flux = np.array(data['Relative_Flux'])
    flux_err = np.array(data['Relative_Flux_err'])

    # Bin the data
    time_binned, flux_binned, fluxerr_binned = bin_time_flux_error(time, flux, flux_err, 12)

    # Calculate mean of normalised data
    mean_dt_flux = np.mean(flux_binned)
    adjustment = mean_dt_flux - (0.9968 if cam == "CMOS" else 0.998)  # Adjust for CMOS or CCD

    # Adjust flux
    dt_flux_adjusted = flux_binned - adjustment
    flux_adjusted = flux - adjustment

    # Normalize time array to be centered around the transit
    time_centered = time - params.t0

    # Initialize the transit model with the centered time array
    m = batman.TransitModel(params, time_binned)
    model_flux = m.light_curve(params)

    return time, flux_adjusted, flux_err, time_binned, dt_flux_adjusted, fluxerr_binned, model_flux


# Process both cameras
time1, flux_adjusted1, flux_err1, time_binned1, dt_flux_adjusted1, fluxerr_binned1, model_flux1 = process_camera(cam1, target)
time2, flux_adjusted2, flux_err2, time_binned2, dt_flux_adjusted2, fluxerr_binned2, model_flux2 = process_camera(cam2, target)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
axes[0].plot(time1, flux_adjusted1, '.', label=f"{cam1} Unbinned", color="grey", alpha=0.5)
axes[0].plot(time_binned1, dt_flux_adjusted1, 'o', label=f"{cam1} 5 min bin", color="red")
axes[0].plot(time_binned1, model_flux1, label=f"{cam1} Transit Model", color="black", linestyle='-')
axes[0].set_xlabel("Time (BJD)")
axes[0].set_ylabel("Relative flux")
# axes[0].legend()
axes[0].set_title(f"{cam1} Data")

axes[1].plot(time2, flux_adjusted2, '.', label=f"{cam2} Unbinned", color="grey", alpha=0.5)
axes[1].plot(time_binned2, dt_flux_adjusted2, 'o', label=f"{cam2} 5 min bin", color="blue")
axes[1].plot(time_binned2, model_flux2, label=f"{cam2} Transit Model", color="black", linestyle='-')
axes[1].set_xlabel("Time (BJD)")
# axes[1].legend()
axes[1].set_title(f"{cam2} Data")

# Adjust layout and show
plt.tight_layout()
plt.show()