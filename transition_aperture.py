import numpy as np
from astropy.io import fits
import os
from calibration_images import reduce_images
import json
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

    r2 = (xx - x)**2 + (yy - y)**2
    aperture_mask = r2 <= r**2

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

from matplotlib.patches import Circle

def show_star_aperture(image, x, y, r):
    """
    Display star cutout with pixel values and aperture overlay
    """

    x = int(round(x))
    y = int(round(y))
    r = int(np.ceil(r))

    # Cutout box
    cutout = image[y-r:y+r+1, x-r:x+r+1]

    plt.figure(figsize=(5,5))
    plt.imshow(cutout, origin='lower', cmap='gray')
    plt.colorbar(label="Counts")

    # Annotate pixel values
    for j in range(cutout.shape[0]):
        for i in range(cutout.shape[1]):
            val = cutout[j, i]
            plt.text(i, j, f"{int(val)}",
                     color='red',
                     ha='center',
                     va='center',
                     fontsize=8)

    # Aperture circle (centered in cutout)
    circ = Circle((r, r), r, edgecolor='cyan',
                  facecolor='none', linewidth=2)
    plt.gca().add_patch(circ)

    plt.title("Star aperture with pixel values")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.tight_layout()
    plt.show()

target_tic = 270187200

idx = np.where(phot_cat['tic_id'] == target_tic)[0]

if len(idx) == 0:
    raise ValueError(f"TIC {target_tic} not found in catalog")

i = idx[0]   # first match

x_star = phot_x[i]
y_star = phot_y[i]
mag_star = phot_cat['Tmag'][i]

print(f"Selected TIC {target_tic}")
print(f"x={x_star:.2f}, y={y_star:.2f}, Tmag={mag_star:.2f}")


show_star_aperture(frame_data, x_star, y_star, r=5)