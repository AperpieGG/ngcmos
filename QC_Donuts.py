#!/usr/bin/env python3

"""
This is a script to run Donuts on a set of images, compute the shifts, and save the results to a JSON file.
It also created a mp4 animation of the images with shifts greater than 0.5 pixels.
"""

import json
import os
from datetime import datetime, timedelta
import numpy as np
from donuts import Donuts
import glob
from matplotlib import pyplot as plt
import warnings
from astropy.io import fits
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from utils import plot_images, utc_to_jd

warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.fromnumeric")
warnings.filterwarnings("ignore", category=UserWarning, module="donuts.image")


def find_current_night_directory(base_path):
    # Get the current date in the format YYYYMMDD
    current_date = datetime.now().strftime("%Y%m%d")
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(base_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return None


def find_first_image_of_each_prefix(directory, save_path):
    # List all items (files and directories) in the given directory
    items = os.listdir(directory)

    # Filter out files with the words "flat," "bias," and "dark" in their names
    filtered_items = [item for item in items if "flat" not in item.lower() and "bias" not in item.lower() and "dark" not in item.lower()]

    # Dictionary to store the first image of each prefix
    first_image_of_each_prefix = {}

    # Words to exclude in the prefix
    exclude_words = ["evening", "morning", "flat", "bias", "dark"]

    # Iterate through filtered items
    for item in filtered_items:
        # Check if any exclude word is in the item
        if any(word in item.lower() for word in exclude_words):
            continue  # Skip this item if any exclude word is found

        # Extract the first 6 letters of the item
        prefix = item[:11]

        # Check if the prefix is already a key in the dictionary
        if prefix not in first_image_of_each_prefix:
            # Update the file path pattern for the given prefix
            pattern = os.path.join(directory, f'{prefix}*.fits')
            # Use glob to find matching files
            matching_files = glob.glob(pattern)
            # Sort the matching files
            matching_files = sorted(matching_files)
            # Check if any matching files were found
            if matching_files:
                first_image_of_each_prefix[prefix] = matching_files[0]

    # Print the first image for each different prefix
    print(f"First image of each prefix in {directory} (excluding those with 'flat', 'bias', 'dark', 'evening', "
          f"and 'morning' in the name):\n")
    for prefix, first_image in first_image_of_each_prefix.items():
        print(f"Prefix: {prefix}, First Image: {first_image}")
        # Assuming you have a run_donuts function defined
        run_donuts(directory, prefix, save_path)

    if not first_image_of_each_prefix:
        print(f"No images found in {directory} with the specified prefix.")


def run_donuts(directory, prefix, save_path):
    path = directory + '/'
    image_names = glob.glob(path + f'{prefix}*.fits')
    image_names = sorted(image_names)

    if not image_names:
        print(f"No images found for prefix: {prefix}")
        return

    reference_image_name = image_names[0]

    # Print some debugging information
    print(f"Using {reference_image_name} as the reference image for prefix: {prefix}\n")

    science_image_names = [f for f in glob.glob(path + f'{prefix}*.fits')[1:]]
    science_image_names = sorted(science_image_names)

    d = Donuts(
        refimage=reference_image_name,
        image_ext=0,
        overscan_width=20,
        prescan_width=20,
        border=64,
        normalise=True,
        exposure='EXPTIME',
        subtract_bkg=True,
        ntiles=32)
    # for each image, compute the x/y translation required
    # to align the images onto the reference image

    x_shifts = []
    y_shifts = []

    for image in science_image_names:
        shift_result = d.measure_shift(image)
        x = shift_result.x
        y = shift_result.y

        if abs(x.value) < 0.5 and abs(y.value) < 0.5:
            print("Image {} with shifts (x, y): {}, {}".format(image[-22:], x.value, y.value))
        elif abs(x.value) >= 0.5 or abs(y.value) >= 0.5:
            print('WARNING: Image {} is not aligned with shifts (x, y): {}, {}'.format(image[-22:], x.value, y.value))
        else:
            pass

        # Append shift values to the lists
        x_shifts.append(x.value)
        y_shifts.append(y.value)

    num_large_shifts = sum(1 for x, y in zip(x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5)
    print("The number of images with shifts greater than 0.5 pixels is: {}".format(num_large_shifts))
    print()

    plot_images()

    save_results(x_shifts, y_shifts, reference_image_name, save_path, prefix, science_image_names)

    time = acquire_header_info(directory, prefix)

    plot_shifts(x_shifts, y_shifts, save_path, prefix, time)

    create_blink_animation(science_image_names, x_shifts, y_shifts, prefix, save_path)


def create_blink_animation(science_image_names, x_shifts, y_shifts, prefix, save_path):
    images_with_large_shift = [image for image, x, y in zip(science_image_names, x_shifts, y_shifts)
                               if abs(x) >= 0.5 or abs(y) >= 0.5]

    if not images_with_large_shift:
        print("No images with shifts greater than 0.5 pixels.")
    else:
        print("Creating animation for images with shifts greater than 0.5 pixels.\n")
        print(f"Number of images with shifts greater than 0.5 pixels: {len(images_with_large_shift)}\n")

        # get the date of the images
        date_obs = [fits.getheader(image)['DATE-OBS'] for image in images_with_large_shift]

        object_name = [fits.getheader(image)['OBJECT'] for image in images_with_large_shift]
        ra = [fits.getheader(image)['TELRA'] for image in images_with_large_shift]
        dec = [fits.getheader(image)['TELDEC'] for image in images_with_large_shift]

        # Default output path with object name and date
        timestamp_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

        # Construct the directory path based on the current date
        base_file_name = f"donuts_{prefix}_{timestamp_yesterday}"

        # Construct the full file path within the "shifts_plots" directory
        gif_file_path = os.path.join(save_path, f"{base_file_name}.mp4")

        fig, ax = plt.subplots(figsize=(8, 8))

        data_image = fits.getdata(images_with_large_shift[0])
        vmin = np.median(data_image[0]) - 0.2 * np.median(data_image[0])
        vmax = np.median(data_image[0]) + 0.2 * np.median(data_image[0])
        norm = Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(data_image, cmap='hot', origin='lower', norm=norm)
        ax.set_xlabel('X-axis [pix]')
        ax.set_ylabel('Y-axis [pix]')
        ax.set_title('QC guiding')
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                            fontsize=10, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.8))
        frame_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, color='white',
                             fontsize=10, verticalalignment='top', horizontalalignment='right',
                             bbox=dict(facecolor='black', alpha=0.8))
        info_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, color='white',
                            fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.8))
        if "TOI" in object_name[0]:
            object_text = ax.text(0.78, 0.02, '', transform=ax.transAxes, color='white',
                                  fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.8))
        elif "TIC" in object_name[0]:
            object_text = ax.text(0.72, 0.02, '', transform=ax.transAxes, color='white',
                                  fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.8))
        else:
            object_text = ax.text(0.70, 0.02, '', transform=ax.transAxes, color='white',
                                  fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.8))

        def update(frame):
            im.set_array(fits.getdata(images_with_large_shift[frame]))
            data_images = fits.getdata(images_with_large_shift[frame])
            vmin = np.median(data_images) - 0.2 * np.median(data_images)
            vmax = np.median(data_images) + 0.2 * np.median(data_images)

            # Update normalization plots
            norm = Normalize(vmin=vmin, vmax=vmax)
            im.set_norm(norm)

            time_text.set_text(f'DATE-OBS: {date_obs[frame]}')
            frame_text.set_text(f'Frame: {frame + 1}')
            object_ra_dec_text = f'RA: {ra[frame]}\nDEC: {dec[frame]}'
            info_text.set_text(object_ra_dec_text)
            object_text.set_text(f'Object: {object_name[frame][:11]}')
            if "TOI" in object_name[frame]:
                # Find the index where "TOI" starts
                toi_index = object_name[frame].find("TOI")
                # Extract "TOI" and the next 5 letters
                toi_and_next_five = object_name[frame][toi_index:toi_index + 9]
                object_text.set_text(f'Object: {toi_and_next_five}')
            elif "TIC" in object_name[frame]:
                # Find the index where "TIC" starts
                tic_index = object_name[frame].find("TIC")
                # Extract "TIC" and the next 5 letters
                tic_and_next_five = object_name[frame][tic_index:tic_index + 13]
                object_text.set_text(f'Object: {tic_and_next_five}')
            else:
                object_text.set_text(f'Object: {object_name[frame][:11]}')

            return [im, time_text, object_text, frame_text, info_text]

        animation = FuncAnimation(fig, update, frames=len(images_with_large_shift), blit=True)
        animation.save(gif_file_path, writer='ffmpeg', fps=5)
        print(f"Animation saved to: {gif_file_path}\n")


def acquire_header_info(directory, prefix):
    path = directory + '/'
    image_names = glob.glob(path + f'{prefix}*.fits')
    image_names = sorted(image_names[1:])
    time_jd = []

    for image in image_names:
        with fits.open(image) as hdulist:
            header = hdulist[0].header
            # Extract the UTC time string from the header
            utc_time_str = header['DATE-OBS']

            # Convert the UTC time string to JD
            jd = utc_to_jd(utc_time_str)

            time_jd.append(jd)

    return time_jd


def plot_shifts(x_shifts, y_shifts, save_path, prefix, time):
    # Plot the shifts with colorbar
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_shifts, y_shifts, c=time, cmap='viridis', label='Shifts for field: {}'.format(prefix), marker='o')
    plt.xlabel('X Shift (pixels)')
    plt.ylabel('Y Shift (pixels)')
    plt.title('Shifts with respect to the ref image')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)  # Add horizontal line at y=0
    plt.axvline(0, color='black', linestyle='-', linewidth=1)  # Add vertical line at x=0
    plt.legend()

    # Set the axes limits to center (0, 0)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Add colorbar
    cbar = plt.colorbar(scatter, label='Time')

    # Get the prev night directory in the format DDMMYYYY
    timestamp_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the directory path based on the current date
    base_file_name = f"donuts_{prefix}_{timestamp_yesterday}"

    # Construct the full file path within the "shifts_plots" directory
    pdf_file_path = os.path.join(save_path, f"{base_file_name}.pdf")

    # Save the figure
    fig.savefig(pdf_file_path, bbox_inches='tight')
    print(f"PDF plot saved to: {pdf_file_path}\n")


def save_results(x_shifts, y_shifts, reference_image_name, save_path, prefix, science_image_names):
    # Get the prev night directory in the format DDMMYYYY
    timestamp_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the base file name
    base_file_name = f"donuts_{prefix}_{timestamp_yesterday}"

    # Construct the full file paths
    json_file_path = os.path.join(save_path, f"{base_file_name}.json")

    num_large_shifts = sum(1 for x, y in zip(x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5)

    # Save the results to the JSON file
    results_data = {
        "Reference Image": reference_image_name,
        "The number of images with shifts greater than +/-0.5 pixels is": {
            "Total Images": len(science_image_names),
            "Number of Images with Large Shifts": num_large_shifts
        },
        "The name of the images with shifts greater than 0.5 pixels is":
            [image for image, x, y in zip(science_image_names, x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5],
        "And shifts of this/them": {
            "X Shifts": [x for x, y in zip(x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5],
            "Y Shifts": [y for x, y in zip(x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5]
        },
        "X Shifts and Y Shifts": list(zip(x_shifts, y_shifts)),
    }

    with open(json_file_path, 'w') as json_file:
        json.dump(results_data, json_file, indent=4)

    print(f"JSON results saved to: {json_file_path}")


def main():
    # First directory
    base_path_1 = '/Users/u5500483/Downloads/DATA_MAC/CMOS/'
    # Second directory
    base_path_2 = '/home/ops/data/'

    # Check if the first directory exists
    if os.path.exists(base_path_1):
        base_path = base_path_1
    else:
        base_path = base_path_2

    save_path = base_path + 'shifts_plots/'
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Find the current night directory
    current_night_directory = find_current_night_directory(base_path)

    if current_night_directory:
        print(f"Current night directory found: {current_night_directory}")
        find_first_image_of_each_prefix(current_night_directory, save_path)
    else:
        print("No current night directory found.")


if __name__ == "__main__":
    main()






