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
filename = 'NG2320-1302_TIC-188620407_S43-20240706072737751.fits'
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

    return np.sum(good), aperture_pixels.size


radius = 5
mag_list = []
count_list = []
fraction_list = []

for xi, yi, mag in zip(phot_x, phot_y, phot_cat['Tmag']):
    xi = int(round(xi))
    yi = int(round(yi))

    n_good, n_total = count_pixels_in_range(frame_data, xi, yi, radius)

    mag_list.append(mag)
    count_list.append(n_good)
    fraction_list.append(100.0 * n_good / n_total)

plt.figure()
plt.scatter(mag_list, fraction_list, s=12)
plt.xlabel("Tmag")
plt.ylabel("Percentage of aperture pixels in range (%)")
plt.gca().invert_xaxis()
plt.show()


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
    plt.xlim(0,12)
    plt.ylim(0,12)
    plt.tight_layout()
    plt.show()


target_tic = 270187200

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
