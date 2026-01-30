#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from astropy.wcs import WCS

from calibration_images import reduce_images
from utils import *


plot_images()

# ======================================================
# USER SETTINGS
# ======================================================
TARGET_TIC = 188620486
R_AP = 5
R_IN = 15
R_OUT = 20
GAIN_CONSTANT = 1.13  # e-/ADU

CATALOG_FILE = "NG2320-1302_catalog_input.fits"
DIRECTORIES_FILE = "directories.json"
GAIN_SPLINE_FILE = "/home/ops/ngcmos/gain_vs_signal_spline.pkl"

# ======================================================
# LOAD DIRECTORIES
# ======================================================
with open(DIRECTORIES_FILE) as f:
    config = json.load(f)

calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# pick first base_path that exists
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break

# ======================================================
# LOAD CATALOG
# ======================================================
phot_cat, _ = get_catalog(CATALOG_FILE, ext=1)

# ======================================================
# LOAD GAIN SPLINE MODEL
# ======================================================
with open(GAIN_SPLINE_FILE, "rb") as f:
    gain_model = pickle.load(f)

print("Loaded spline gain model.")

# ======================================================
# LIST FITS FILES
# ======================================================
files = [f for f in os.listdir(base_path) if f.endswith(".fits") and f.startswith("NG2320-1302_TIC-188620407")]
files.sort()

print(f"Found {len(files)} FITS files")

# ======================================================
# SIMPLE APERTURE PHOTOMETRY FUNCTION
# ======================================================
def aperture_photometry(data, x, y, r_ap=5, r_in=15, r_out=20):
    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]
    r2 = (xx - x) ** 2 + (yy - y) ** 2

    ap_mask = r2 <= r_ap ** 2
    sky_mask = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)

    ap_pixels = data[ap_mask]
    sky_pixels = data[sky_mask]

    sky_median = np.median(sky_pixels)
    net_flux = np.sum(ap_pixels) - sky_median * ap_pixels.size

    return net_flux, sky_median, ap_pixels.size

# ======================================================
# GAIN-CORRECTED PHOTOMETRY
# ======================================================
def aperture_photometry_gain_corrected(data, x, y, r_ap=5, r_in=15, r_out=20, gain_model=None):
    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]
    r2 = (xx - x) ** 2 + (yy - y) ** 2

    ap_mask = r2 <= r_ap ** 2
    sky_mask = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)

    ap_pixels = data[ap_mask]
    sky_pixels = data[sky_mask]

    # convert sky pixels to electrons
    sky_e = sky_pixels * gain_model(sky_pixels)
    sky_e_per_pix = np.median(sky_e)

    # convert aperture pixels to electrons
    ap_e = ap_pixels * gain_model(ap_pixels)
    net_flux_e = np.sum(ap_e) - sky_e_per_pix * ap_pixels.size

    return net_flux_e, sky_e_per_pix, ap_pixels.size

# ======================================================
# LOOP OVER FILES
# ======================================================
flux_constant_list = []
flux_spline_list = []
sky_constant_list = []
sky_spline_list = []
n_pix_list = []
frame_times = []

for i, filename in enumerate(files):
    print(f"\nProcessing {filename}")

    reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])
    frame_data = reduced_data[0]
    header = reduced_header[0]

    wcs = WCS(header)
    x_all, y_all = wcs.all_world2pix(phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1)
    idx = np.where(phot_cat['tic_id'] == TARGET_TIC)[0]
    if len(idx) == 0:
        print(f"TIC {TARGET_TIC} not found in catalog for {filename}")
        continue
    k = idx[0]
    x_star = x_all[k]
    y_star = y_all[k]

    # -----------------------------
    # Constant Gain Photometry
    # -----------------------------
    flux_c, sky_c, n_pix = aperture_photometry(frame_data, x_star, y_star, R_AP, R_IN, R_OUT)
    flux_c *= GAIN_CONSTANT
    sky_c *= GAIN_CONSTANT

    # -----------------------------
    # Signal-dependent Gain Photometry
    # -----------------------------
    flux_s, sky_s, _ = aperture_photometry_gain_corrected(frame_data, x_star, y_star,
                                                          R_AP, R_IN, R_OUT, gain_model=gain_model)

    flux_constant_list.append(flux_c)
    flux_spline_list.append(flux_s)
    sky_constant_list.append(sky_c)
    sky_spline_list.append(sky_s)
    n_pix_list.append(n_pix)
    frame_times.append(i)

# ======================================================
# PLOT FLUX VS TIME
# ======================================================
plot_images()

plt.figure(figsize=(10,6))
plt.plot(frame_times, flux_constant_list, 'o-', label='Constant Gain')
plt.plot(frame_times, flux_spline_list, 'o-', label='Signal-Dependent Gain')
plt.xlabel("Frame index")
plt.ylabel("Net Flux (electrons)")
plt.title(f"Aperture Photometry: TIC {TARGET_TIC}")
plt.legend()
plt.tight_layout()
plt.show()