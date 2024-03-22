#!/usr/bin/env python
import json
import os
import argparse
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from utils import (plot_images, find_current_night_directory, get_phot_files, read_phot_file,
                   bin_time_flux_error)
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval


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


def get_image_data(frame_id, image_directory):
    """
    Get the image data corresponding to the given frame_id.

    Parameters:
        frame_id (str): The frame_id of the image.
        image_directory (str): The directory where the image files are stored.

    Returns:
        numpy.ndarray or None: The image data if the image exists, otherwise None.
    """
    # Construct the path to the image file using the frame_id
    image_path = os.path.join(image_directory, frame_id)

    # Check if the image file exists
    if os.path.exists(image_path):
        # Open the image file
        try:
            image_data = fits.getdata(image_path)
            return image_data
        except Exception as e:
            print(f"Error opening image file {image_path}: {e}")
            return None
    else:
        print(f"Image file {image_path} not found.")
        return None


def plot_lc(table, gaia_id_to_plot, bin_size, image_directory=""):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]

    if len(gaia_id_data) == 0:
        print(f"Gaia ID {gaia_id_to_plot} not found in the current photometry file.")
        return
    tmag = gaia_id_data['Tmag'][0]
    jd_mid = gaia_id_data['jd_mid']
    x = gaia_id_data['x'][0]
    y = gaia_id_data['y'][0]

    # # Extract fluxes and errors based on Tmag
    # if tmag < 11:
    #     fluxes = gaia_id_data['flux_5']
    #     fluxerrs = gaia_id_data['fluxerr_5']
    #     sky = gaia_id_data['flux_w_sky_5'] - gaia_id_data['flux_5']
    #     skyerrs = np.sqrt(gaia_id_data['fluxerr_5'] ** 2 + gaia_id_data['fluxerr_w_sky_5'] ** 2)
    # elif 12 > tmag >= 11:
    #     fluxes = gaia_id_data['flux_4']
    #     fluxerrs = gaia_id_data['fluxerr_4']
    #     sky = gaia_id_data['flux_w_sky_4'] - gaia_id_data['flux_4']
    #     skyerrs = np.sqrt(gaia_id_data['fluxerr_4'] ** 2 + gaia_id_data['fluxerr_w_sky_4'] ** 2)
    # else:
    #     fluxes = gaia_id_data['flux_3']
    #     fluxerrs = gaia_id_data['fluxerr_3']
    #     sky = gaia_id_data['flux_w_sky_3'] - gaia_id_data['flux_3']
    #     skyerrs = np.sqrt(gaia_id_data['fluxerr_3'] ** 2 + gaia_id_data['fluxerr_w_sky_3'] ** 2)

    fluxes = gaia_id_data['flux_6']
    fluxerrs = gaia_id_data['fluxerr_6']
    sky = gaia_id_data['flux_w_sky_6'] - gaia_id_data['flux_6']
    skyerrs = np.sqrt(gaia_id_data['fluxerr_6'] ** 2 + gaia_id_data['fluxerr_w_sky_6'] ** 2)

    # Bin flux data
    jd_mid_binned, fluxes_binned, fluxerrs_binned = bin_time_flux_error(jd_mid, fluxes, fluxerrs, bin_size)
    # Bin sky data using the same binned jd_mid as the flux data
    _, sky_binned, skyerrs_binned = bin_time_flux_error(jd_mid, sky, skyerrs, bin_size)

    # Define the size of the figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    airmass = []
    # take data for the first frame_id
    image_data = get_image_data(gaia_id_data['frame_id'][0], image_directory)

    # Get airmass for each frame_id
    for frame_id in gaia_id_data['frame_id']:
        image_header = fits.getheader(os.path.join(image_directory, frame_id))
        airmass.append(round(image_header['AIRMASS'], 2))
    print(f"The star has GAIA id: {gaia_id_to_plot}")
    print(len(airmass))
    print(len(jd_mid_binned))

    # Plot the image data
    if image_data is not None:
        # Define the size of the region around the star
        radius = 30  # pixels

        # Define the limits for the region around the star
        x_min = max(int(x - radius), 0)
        x_max = min(int(x + radius), image_data.shape[1])
        y_min = max(int(y - radius), 0)
        y_max = min(int(y + radius), image_data.shape[0])

        # Crop the image data to the defined region
        cropped_image_data = image_data[y_min:y_max, x_min:x_max]
        # Normalize the cropped image data using zscale
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(cropped_image_data)
        normalized_image_data = np.clip((cropped_image_data - vmin) / (vmax - vmin), 0, 1)
        # Plot the normalized cropped image
        extent = [x - radius, x + radius, y - radius, y + radius]
        im = axs[2].imshow(normalized_image_data, cmap='gray', origin='lower', extent=extent)
        axs[2].set_xlabel('X')
        axs[2].set_ylabel('Y')

        # Draw a circle around the target star
        # if tmag < 10.5:
        #     circle_radii = [6]
        # elif 10.5 <= tmag < 11:
        #     circle_radii = [5]
        # elif 12 > tmag >= 11:
        #     circle_radii = [4]
        # else:
        #     circle_radii = [3]

        circle_radii = [6]

        for radius in circle_radii:
            circle = Circle((x, y), radius=radius, edgecolor='lime', facecolor='none', lw=1)
            axs[2].add_patch(circle)
        annulus = Circle((x, y), radius=15, edgecolor='lime', facecolor='none', lw=1, linestyle='dashed')
        dannulus = Circle((x, y), radius=20, edgecolor='lime', facecolor='none', lw=1, linestyle='dashed')
        axs[2].add_patch(annulus)
        axs[2].add_patch(dannulus)

        # Create legend labels
        legend_labels = [f'Aperture, {radius} pix)' for radius in circle_radii]
        legend_labels.append('Annulus, 15-20 pix')
        axs[2].legend(legend_labels, loc='upper left', bbox_to_anchor=(1.01, 1.0))

        # Plot jd_mid vs flux
        axs[0].errorbar(jd_mid_binned, fluxes_binned, yerr=fluxerrs_binned, fmt='o', color='black', label='Raw Flux')
        axs[0].set_title(f'LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f})')
        axs[0].set_ylabel('Flux [e-]')
        axs[0].legend()

        ax2 = axs[0].twiny()
        ax2.set_xlim(min(airmass), max(airmass))  # Set the limits based on airmass values
        # ax2.invert_yaxis()
        ax2.set_xlabel('Airmass')

        ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins=len(axs[0].get_xticks()), prune='both'))
        # ax2.plot(jd_mid_binned, airmass, 'o', color='red', label='Airmass')

        # Plot jd_mid vs sky
        axs[1].errorbar(jd_mid_binned, sky_binned, yerr=skyerrs_binned, fmt='o', color='red', label='Sky')
        axs[1].set_ylabel('Flux [e-]')
        axs[1].set_xlabel('MJD [days]')
        axs[1].legend()
        plt.tight_layout()
        plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('-gaia_ids', nargs='+', type=int, help='List of Gaia IDs of the stars to plot', required=True)
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()
    bin_size = args.bin

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        # Check if gaia_id exists in the current photometry file
        for gaia_id in args.gaia_ids:
            if gaia_id in phot_table['gaia_id']:
                print('Found star in photometry file:', phot_file)
                plot_lc(phot_table, gaia_id, bin_size, image_directory=current_night_directory)
                break
        else:
            print(f"Gaia IDs {args.gaia_ids} not found in {phot_file}")
    else:
        print(f"Gaia ID {args.gaia_id} not found in any photometry file.")


if __name__ == "__main__":
    main()
