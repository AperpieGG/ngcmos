#!/usr/bin/env python
import json
import os
import argparse
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from utils import plot_images, get_phot_files, read_phot_file, bin_time_flux_error, remove_outliers
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval


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
    image_path_fits = os.path.join(image_directory, frame_id)
    image_path_bz2 = os.path.join(image_directory, frame_id + '.bz2')

    # Check if the image file with .fits extension exists
    if os.path.exists(image_path_fits):
        try:
            # Open the image file
            image_data = fits.getdata(image_path_fits)
            return image_data
        except Exception as e:
            print(f"Error opening image file {image_path_fits}: {e}")
            return None

    # Check if the image file with .bz2 extension exists
    elif os.path.exists(image_path_bz2):
        try:
            # Open the image file
            image_data = fits.getdata(image_path_bz2)
            return image_data
        except Exception as e:
            print(f"Error opening image file {image_path_bz2}: {e}")
            return None

    # If neither .fits nor .bz2 file exists
    else:
        print(f"Image file {frame_id} not found.")
        return None


def plot_lc(table, tic_id_to_plot, bin_size, aperture, image_directory=""):
    # Select rows with the specified TIC ID
    tic_id_data = table[table['tic_id'] == tic_id_to_plot]

    if len(tic_id_data) == 0:
        print(f"TIC ID {tic_id_to_plot} not found in the current photometry file.")
        return
    tmag = tic_id_data['Tmag'][0]
    jd_mid = tic_id_data['jd_bary']
    x = tic_id_data['x'][0]
    y = tic_id_data['y'][0]
    print(f'Stars position as re in x: {x}, y: {y}')
    fluxes = tic_id_data[f'flux_{aperture}']
    fluxerrs = tic_id_data[f'fluxerr_{aperture}']
    sky = tic_id_data[f'flux_w_sky_{aperture}'] - tic_id_data[f'flux_{aperture}']
    skyerrs = np.sqrt(tic_id_data[f'fluxerr_{aperture}'] ** 2 + tic_id_data[f'fluxerr_w_sky_{aperture}'] ** 2)
    airmass = tic_id_data['airmass']

    # remove outliers
    time_clipped, fluxes_clipped, fluxerrs_clipped, sky_clipped, skyerrs_clipped = (
        remove_outliers(jd_mid, fluxes, fluxerrs, sky, skyerrs))
    # Bin flux data
    jd_mid_binned, fluxes_binned, fluxerrs_binned = (
        bin_time_flux_error(time_clipped, fluxes_clipped, fluxerrs_clipped, bin_size))
    # Bin sky data using the same binned jd_mid as the flux data
    _, sky_binned, skyerrs_binned = bin_time_flux_error(time_clipped, sky_clipped, skyerrs_clipped, bin_size)

    # Define the size of the figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # take data for the first frame_id
    image_data = get_image_data(tic_id_data['frame_id'][0], image_directory)

    print(f"The star has TIC id: {tic_id_to_plot}")
    print(f"Using the frame_id: {tic_id_data['frame_id'][0]}")

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

        circle_radii = [aperture]

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
        axs[0].set_title(f'Raw lightcurve for TIC ID {tic_id_to_plot} (Tmag = {tmag:.2f})')
        axs[0].set_ylabel('Flux [ADU]')
        axs[0].legend()

        ax2 = axs[0].twiny()
        ax2.set_xlim(axs[0].get_xlim())

        unique_jd_mid = np.unique(jd_mid)
        unique_airmass = [airmass[np.where(jd_mid == jd)[0][0]] for jd in unique_jd_mid]
        airmass_interp = interp1d(unique_jd_mid, unique_airmass, fill_value="extrapolate")
        tick_locations = axs[0].get_xticks()
        binned_airmass = airmass_interp(tick_locations)

        binned_airmass = np.clip(binned_airmass, min(unique_airmass), max(unique_airmass))

        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels([f'{am:.2f}' for am in binned_airmass])
        ax2.set_xlabel('Airmass')

        # Plot jd_mid vs sky
        axs[1].errorbar(jd_mid_binned, sky_binned, yerr=skyerrs_binned, fmt='o', color='red', label='Sky')
        axs[1].set_ylabel('Flux [ADU]')
        axs[1].set_xlabel('BJD [days]')
        axs[1].legend()
        plt.tight_layout()
        plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('tic_id', type=int, help='The TIC ID of the star to plot')
    parser.add_argument('aperture', type=int,
                        help='The aperture size for photometry (i.e. 1-6)')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()
    bin_size = args.bin
    aperture = args.aperture

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = '.'

    print(f"Current night directory: {current_night_directory}")

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        # Check if tic_id exists in the current photometry file
        if args.tic_id in phot_table['tic_id']:
            print('Found star in photometry file:', phot_file)
            plot_lc(phot_table, args.tic_id, bin_size, aperture, image_directory=current_night_directory)
            break  # Stop looping if tic_id is found
        else:
            print(f"TIC ID {args.tic_id} not found in {phot_file}")

    else:
        print(f"TIC ID {args.tic_id} not found in any photometry file.")


if __name__ == "__main__":
    main()
