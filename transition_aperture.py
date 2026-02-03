import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
from calibration_images import reduce_images
import json
from matplotlib.patches import Circle
from utils import *

# read the image
# read the catalog
# set aperture to 5 and within measure how many pixels are 1800 -+ 200

plot_images()


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
filename = 'NG2320-1302_TIC-188620407_S43-20240706092807977.fits'
reduced_data, reduced_header, _ = reduce_images(base_path, out_path, [filename])

reduced_data_dict = {
    filename: (data, header)
    for data, header in zip(reduced_data, reduced_header)
}

frame_data, frame_hdr = reduced_data_dict[filename]

phot_cat, _ = get_catalog(
    os.path.join(directory, "NG2320-1302_catalog_input.fits"),
    ext=1
)
phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_cat['ra_deg_corr'], phot_cat['dec_deg_corr'], 1)


def count_pixels_in_range(data, x, y, r, low=1600, high=2000):
    """
    Counts how many pixels inside a circular aperture
    fall between [low, high].
    """
    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]

    r2 = (xx - x) ** 2 + (yy - y) ** 2
    aperture_mask = r2 <= r ** 2

    aperture_pixels = data[aperture_mask]
    good = (aperture_pixels >= low) & (aperture_pixels <= high)

    n_good = np.sum(good)

    if n_good > 0:
        max_val = np.max(aperture_pixels[good])
    else:
        max_val = None

    return n_good, aperture_pixels.size, max_val


radius = 5
mag_list = []
count_list = []
pixel_number_list = []


for xi, yi, mag, tic in zip(phot_x, phot_y, phot_cat['Tmag'], phot_cat['TIC_ID']):
    xi = int(round(xi))
    yi = int(round(yi))

    n_good, n_total, max_val = count_pixels_in_range(frame_data, xi, yi, radius)

    mag_list.append(mag)
    count_list.append(n_total)
    pixel_number_list.append(n_good)


plt.figure(figsize=(8, 6))
plt.scatter(mag_list, pixel_number_list, s=12, c='blue', alpha=0.7)
plt.grid(alpha=0.3)
plt.xlabel("Tmag")
plt.ylabel("#/78 of pixels in transition")
plt.gca().invert_xaxis()
# save figure
plt.tight_layout()
plt.savefig('transition_pixels.pdf', dpi=300)
plt.show()

mag_edges = np.arange(6, 17, 1)

print("\nTransition-pixel statistics per magnitude bin:")

for mmin, mmax in zip(mag_edges[:-1], mag_edges[1:]):

    idx = [(m >= mmin and m < mmax) for m in mag_list]

    mags_bin = np.array(mag_list)[idx]
    ntrans_bin = np.array(pixel_number_list)[idx]

    total_stars = len(mags_bin)
    affected = np.sum(ntrans_bin > 0)

    if total_stars > 0:
        avg_pixels = np.mean(ntrans_bin)
    else:
        avg_pixels = 0.0

    print(
        f"{mmin:.0f}-{mmax:.0f} mag : "
        f"{affected} / {total_stars} stars with transition pixels | "
        f"average transition pixels = {avg_pixels:.2f}"
    )


# here plotting scatter of pixels value vs mags for star that have max pixel value transiting pixels
def top_n_pixels_in_aperture(data, x, y, r, n=4):
    """
    Returns the top-n highest pixel values inside a circular aperture.
    """
    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]
    r2 = (xx - x)**2 + (yy - y)**2
    mask = r2 <= r**2

    pixels = data[mask]
    pixels_sorted = np.sort(pixels)[::-1]   # descending
    return pixels_sorted[:n]


def max_pixel_in_aperture(data, x, y, r):
    """
    Returns the maximum pixel value inside a circular aperture of radius r.
    """
    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]

    r2 = (xx - x)**2 + (yy - y)**2
    aperture_mask = r2 <= r**2

    aperture_pixels = data[aperture_mask]
    max_val = np.max(aperture_pixels)

    return max_val


# Lists for red stars (in transition)
tmag_red = []
maxpix_red = []

# Lists for blue stars (not in transition)
tmag_blue = []
maxpix_blue = []
tic_blue = []

# to save in json
red_tics = []
blue_tics = []


radius = 5
low, high = 1600, 2000

for xi, yi, mag, tic in zip(phot_x, phot_y, phot_cat['Tmag'], phot_cat['TIC_ID']):
    xi = int(round(xi))
    yi = int(round(yi))

    # maximum pixel in the aperture
    max_val = max_pixel_in_aperture(frame_data, xi, yi, radius)

    # split stars into red/blue
    if low <= max_val <= high:
        tmag_red.append(mag)
        maxpix_red.append(max_val)
        red_tics.append(str(tic))
        print(f"TIC {tic} | Tmag={mag:.2f} | highest pixel = {max_val:.1f} (transition)")
    else:
        tmag_blue.append(mag)
        maxpix_blue.append(max_val)
        tic_blue.append(tic)

plt.figure(figsize=(8, 6))

# Blue: rest of stars
plt.scatter(tmag_blue, maxpix_blue, s=20, c='blue', alpha=0.7, label='Rest of sample')

# Red: stars in transition
plt.scatter(tmag_red, maxpix_red, s=20, c='red', alpha=0.7, label='Transition pixels')

plt.xlabel("Tmag")
plt.ylabel("Highest pixel value in aperture (ADU)")
plt.gca().invert_xaxis()  # brighter stars left
plt.grid(alpha=0.3)
plt.xlim(11.5, 13)
plt.ylim(0, 4000)
# plt.legend()
plt.tight_layout()
plt.savefig('transition_pixels_magnitude.pdf', dpi=300)
plt.show()

print("\nBlue stars (not in transition) with 12 ≤ Tmag < 13\n"
      "and 2nd–4th brightest pixels also NOT in transition:\n")

for xi, yi, mag, tic in zip(phot_x, phot_y,
                            phot_cat['Tmag'],
                            phot_cat['TIC_ID']):

    xi = int(round(xi))
    yi = int(round(yi))

    max_val = max_pixel_in_aperture(frame_data, xi, yi, radius)

    # must already be blue (max pixel not in transition)
    if not (low <= max_val <= high) and (12 <= mag < 13):

        top4 = top_n_pixels_in_aperture(frame_data, xi, yi, radius, n=4)

        second, third, fourth = top4[1], top4[2], top4[3]

        if not (low <= second <= high or
                low <= third  <= high or
                low <= fourth <= high):
            blue_tics.append(str(tic))

            print(
                f"TIC {tic} | Tmag={mag:.2f} | "
                f"max={max_val:.1f}, "
                f"2nd={second:.1f}, 3rd={third:.1f}, 4th={fourth:.1f} "
                f"(clean)"
            )

# save clean blue tic ids and red tic ids that have max pixel value in transition
output = {
    "red_stars": sorted(set(red_tics)),
    "blue_stars": sorted(set(blue_tics))
}

with open("classified_stars_tic_only.json", "w") as f:
    json.dump(output, f, indent=4)

print("Saved classified_stars_tic_only.json")


def show_star_aperture(frame_data, x_star, y_star, r=5):
    """
    Display a star cutout with counts, median ± 2*RMS color scaling,
    and a circular aperture overlaid. No pixels are masked.

    Parameters
    ----------
    frame_data : 2D array
        The FITS image data.
    x_star, y_star : float
        Pixel coordinates of the star.
    r : int
        Radius of the aperture in pixels.
    """
    ny, nx = frame_data.shape

    # Define square cutout around the star
    x_min = int(max(x_star - r - 1, 0))
    x_max = int(min(x_star + r + 2, nx))
    y_min = int(max(y_star - r - 1, 0))
    y_max = int(min(y_star + r + 2, ny))

    sub_image = frame_data[y_min:y_max, x_min:x_max]

    # Compute mean and RMS for color scaling
    mean_val = np.mean(sub_image)
    rms_val = np.std(sub_image)
    vmin = mean_val - 2 * rms_val
    vmax = mean_val + 2 * rms_val

    plt.figure(figsize=(6, 6))
    im = plt.imshow(sub_image, origin='lower', cmap='gray', vmin=-2000, vmax=30000)
    plt.colorbar(im, label='Counts')

    # Overlay aperture circle
    # Ensure cutout size is odd
    half_size = r
    x_min = int(max(np.floor(x_star - half_size), 0))
    x_max = int(min(np.ceil(x_star + half_size + 1), nx))
    y_min = int(max(np.floor(y_star - half_size), 0))
    y_max = int(min(np.ceil(y_star + half_size + 1), ny))

    sub_image = frame_data[y_min:y_max, x_min:x_max]

    # Star position within cutout (float!)
    x_center = x_star - x_min
    y_center = y_star - y_min

    # Overlay aperture circle at exact center
    circ = Circle((x_center, y_center), r, edgecolor='cyan', facecolor='none', linewidth=2)
    plt.gca().add_patch(circ)
    plt.gca().add_patch(circ)
    plt.gca().add_patch(circ)

    # Annotate pixel values
    for j in range(sub_image.shape[0]):
        for i in range(sub_image.shape[1]):
            plt.text(i + 1, j + 1, f"{int(sub_image[j, i])}",
                     color='green', ha='center', va='center', fontsize=10)

    plt.title(f"Star at x={x_star:.1f}, y={y_star:.1f}, r={r}px aperture")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.xlim(0.5, 12.5)
    plt.ylim(0.5, 12.5)
    plt.tight_layout()
    plt.show()


target_tic = 5796211

idx = np.where(phot_cat['tic_id'] == target_tic)[0]

if len(idx) == 0:
    raise ValueError(f"TIC {target_tic} not found in catalog")

i = idx[0]  # first match

x_star = phot_x[i]
y_star = phot_y[i]
mag_star = phot_cat['Tmag'][i]

print(f"Selected TIC {target_tic}")
print(f"x={x_star:.2f}, y={y_star:.2f}, Tmag={mag_star:.2f}")

show_star_aperture(frame_data, x_star, y_star, r=5)


# Define magnitude bins (e.g., 9-10, 10-11, ..., 15-16)
# mag_bins = np.arange(9, 17, 1)  # adjust max mag as needed
# bin_centers = mag_bins[:-1] + 0.5
#
# print("Magnitude bin | Average % of pixels in range")
# print("-------------------------------------------")
#
# for low, high in zip(mag_bins[:-1], mag_bins[1:]):
#     # select stars in this magnitude bin
#     mask = (np.array(mag_list) >= low) & (np.array(mag_list) < high)
#     if np.any(mask):
#         avg_fraction = np.mean(np.array(fraction_list)[mask])
#         print(f"{low:.0f}-{high:.0f}       | {avg_fraction:.2f}%")
#     else:
#         print(f"{low:.0f}-{high:.0f}       | No stars in this bin")


# ============================================================
# Simple Aperture Photometry (SEP-style)
# ============================================================

def aperture_photometry(data, x, y, r_ap=5, r_in=15, r_out=20):
    """
    Perform simple circular aperture photometry with background subtraction.

    Parameters
    ----------
    data : 2D array
        Image data
    x, y : float
        Star position in pixel coordinates
    r_ap : float
        Aperture radius
    r_in, r_out : float
        Inner/outer radius of background annulus

    Returns
    -------
    net_flux : float
        Background-subtracted flux (ADU)
    sky_median : float
        Background level per pixel
    n_ap_pix : int
        Number of aperture pixels
    """

    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]

    r2 = (xx - x) ** 2 + (yy - y) ** 2

    aperture_mask = r2 <= r_ap ** 2
    annulus_mask = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)

    ap_pixels = data[aperture_mask]
    sky_pixels = data[annulus_mask]

    sky_median = np.median(sky_pixels)

    flux_raw = np.sum(ap_pixels)
    flux_sky = sky_median * ap_pixels.size

    net_flux = flux_raw - flux_sky

    return net_flux, sky_median, ap_pixels.size


flux, sky, n_pix = aperture_photometry(
    frame_data,
    x_star,
    y_star,
    r_ap=5,
    r_in=15,
    r_out=20
)
gain_stable = 1.13  # e-/ADU
print("\n--- Aperture Photometry ---")
print(f"TIC {target_tic}")
print(f"Aperture radius = 5 px")
print(f"Pixels in aperture = {n_pix}")
print(f"Background (median) = {sky * gain_stable:.2f} e/pix")
print(f"Net flux = {flux * gain_stable:.2f} e-")

import pickle

path = '/home/ops/ngcmos/'
with open(path + "gain_vs_signal_spline.pkl", "rb") as f:
    gain_model = pickle.load(f)


def aperture_photometry_gain_corrected(
        data, x, y,
        r_ap=5, r_in=15, r_out=20,
        gain_model=None
):
    """
    Aperture photometry with signal-dependent gain correction.

    Returns:
    --------
    net_flux_e : float
        Net flux in electrons
    sky_e_per_pix : float
        Sky background in electrons per pixel
    n_pix : int
        Number of aperture pixels
    """

    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]

    r2 = (xx - x) ** 2 + (yy - y) ** 2

    ap_mask = r2 <= r_ap ** 2
    sky_mask = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)

    ap_pixels = data[ap_mask]
    sky_pixels = data[sky_mask]

    # --------------------------------
    # Convert SKY pixels → electrons
    # --------------------------------
    sky_gains = gain_model(sky_pixels)
    sky_pixels_e = sky_pixels * sky_gains

    sky_e_per_pix = np.median(sky_pixels_e)

    # --------------------------------
    # Convert APERTURE pixels → electrons
    # --------------------------------
    ap_gains = gain_model(ap_pixels)
    ap_pixels_e = ap_pixels * ap_gains

    total_aperture_e = np.sum(ap_pixels_e)
    total_sky_e = sky_e_per_pix * ap_pixels.size

    net_flux_e = total_aperture_e - total_sky_e

    return net_flux_e, sky_e_per_pix, ap_pixels.size


net_flux_e, sky_e_pix, n_pix = aperture_photometry_gain_corrected(
    frame_data,
    x_star,
    y_star,
    r_ap=5,
    r_in=15,
    r_out=20,
    gain_model=gain_model
)

print("\n--- Gain-Corrected Aperture Photometry ---")
print(f"TIC {target_tic}")
print(f"Aperture radius = 5 px")
print(f"Pixels in aperture = {n_pix}")
print(f"Sky background = {sky_e_pix:.2f} e-/pix")
print(f"Net flux = {net_flux_e:.2f} e-")
