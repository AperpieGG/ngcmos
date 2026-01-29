import numpy as np
from astropy.io import fits
import os
from calibration_images import reduce_images
import json
from utils import *
# read the image
# read the catalog
# set aperture to 5 and within measure how many pixels are 1800 -+ 200


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


directory = '.'
filename = 'NG2320-1302_TIC-188620407_S43-20240706072737751.fits'
reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])

reduced_data_dict = {
    filename: (data, header)
    for data, header in zip(reduced_data, reduced_header)
}

frame_data, frame_hdr = reduced_data_dict[filename]

phot_cat, _ = get_catalog(os.path.join(directory, "NG2320-1302_catalog_input.fits", ext=1))
phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1)


def count_pixels_in_range(image, x, y, r, vmin, vmax):
    """
    Count how many pixels inside a circular aperture (x,y,r)
    fall within [vmin, vmax].

    Returns
    -------
    n_total : int
        Total pixels in aperture
    n_in_range : int
        Pixels within value range
    """

    ny, nx = image.shape
    xi = int(round(x))
    yi = int(round(y))

    # Bounding box
    x0 = max(0, xi - r)
    x1 = min(nx, xi + r + 1)
    y0 = max(0, yi - r)
    y1 = min(ny, yi + r + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr2 = (xx - x)**2 + (yy - y)**2
    aperture_mask = rr2 <= r**2

    pixels = image[y0:y1, x0:x1][aperture_mask]

    n_total = pixels.size
    n_in_range = np.sum((pixels >= vmin) & (pixels <= vmax))

    return n_total, n_in_range


AP_RADIUS = 5
TARGET_VAL = 1800
TOL = 200

vmin = TARGET_VAL - TOL
vmax = TARGET_VAL + TOL

for i in range(len(phot_x)):
    x = phot_x[i]
    y = phot_y[i]

    n_total, n_good = count_pixels_in_range(
        frame_data,
        x, y,
        AP_RADIUS,
        vmin, vmax
    )

    tic = phot_cat['tic_id'][i]
    mag = phot_cat['Tmag'][i]

    print(
        f"TIC {tic:>12} | Tmag={mag:5.2f} | "
        f"{n_good}/{n_total} pixels in [{vmin},{vmax}]"
    )