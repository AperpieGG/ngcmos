#!/usr/bin/env python3
from collections import defaultdict
from datetime import datetime, timedelta
import glob
import os
from astropy.io import fits
from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval, BaseInterval
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_images():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


def find_fits(file_path):
    total_files = sorted(glob.glob(file_path + '*.fits'))
    print(f"Initial images found: {len(total_files)}")
    filtered_files = [item for item in total_files if
                      "flat" not in item.lower() and "bias" not in item.lower() and "dark" not in item.lower()]
    print(f"Exclude flats, bias and darks, filtered images found: {len(filtered_files)}")
    raw_images = sorted(filtered_files[::6])
    return raw_images


def read_fits(file_path):
    images_dt = []
    for file_path in file_path:
        with fits.open(file_path) as hdu_list:
            image_data = hdu_list[0].data
            date_obs = hdu_list[0].header.get('DATE-OBS', 'N/A')
            tel_ra = hdu_list[0].header.get('TELRA', 'N/A')
            tel_dec = hdu_list[0].header.get('TELDEC', 'N/A')
            target_id = hdu_list[0].header.get('OBJECT', 'N/A')
            images_dt.append((image_data, date_obs, tel_ra, tel_dec, target_id))
    return images_dt


def register_images(images):
    return images


def add_text_elements(ax, images):
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                        fontsize=10, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6))

    frame_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, color='white',
                         fontsize=10, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(facecolor='black', alpha=0.8))

    info_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, color='white',
                        fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.6))

    if "TOI" in images[0][4]:
        object_text = ax.text(0.76, 0.02, '', transform=ax.transAxes, color='white',
                              fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.6))
    elif "TIC" in images[0][4]:
        object_text = ax.text(0.70, 0.02, '', transform=ax.transAxes, color='white',
                              fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.6))
    else:
        object_text = ax.text(0.70, 0.02, '', transform=ax.transAxes, color='white',
                              fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.6))

    return time_text, frame_text, info_text, object_text


def create_blink_animation(images, save_path):
    # Default output path with object name and date
    object_name = images[0][4][:11]
    timestamp_yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    output_name = f"control_{object_name}_{timestamp_yesterday}.gif"
    output_path = os.path.join(save_path, output_name)

    # Create the base path directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # Two subplots side by side

    zscale_interval = ZScaleInterval()

    vmin = 100  # Adjust this value based on your data
    vmax = 2000  # Adjust this value based on your data

    norm = ImageNormalize(vmin=vmin, vmax=vmax)

    # Plot for full frame data
    im1 = ax1.imshow(images[0][0][450:550, 600:700], cmap='hot', origin='lower', norm=norm)
    ax1.set_xlabel('X-axis [pix]')
    ax1.set_ylabel('Y-axis [pix]')
    ax1.set_title('Zoom in Image')

    # Plot for cropped data
    im2 = ax2.imshow(images[0][0], cmap='hot', origin='lower', norm=norm)
    ax2.set_xlabel('X-axis [pix]')
    ax2.set_ylabel('Y-axis [pix]')
    ax2.set_title('Full frame Image')

    # Add text elements to both axes
    time_text1, frame_text1, info_text1, object_text1 = add_text_elements(ax1, images)
    time_text2, frame_text2, info_text2, object_text2 = add_text_elements(ax2, images)

    def update(frame):
        # Update for cropped data
        im1.set_array(images[frame][0][450:550, 600:700])
        # Update for full frame data
        im2.set_array(images[frame][0])

        time_text1.set_text(f'DATE-OBS: {images[frame][1]}')
        frame_text1.set_text(f'Frame: {frame + 1}')
        info_text1.set_text(f'RA: {images[frame][2]}, DEC: {images[frame][3]}')
        object_text1.set_text(f'Object: {images[frame][4][:11]}')

        time_text2.set_text(f'DATE-OBS: {images[frame][1]}')
        frame_text2.set_text(f'Frame: {frame + 1}')
        info_text2.set_text(f'RA: {images[frame][2]}, DEC: {images[frame][3]}')
        object_text2.set_text(f'Object: {images[frame][4][:11]}')

        if "TOI" in images[frame][4]:
            toi_index = images[frame][4].find("TOI")
            toi_and_next_five = images[frame][4][toi_index:toi_index + 9]
            object_text1.set_text(f'Object: {toi_and_next_five}')
            object_text2.set_text(f'Object: {toi_and_next_five}')
        elif "TIC" in images[frame][4]:
            tic_index = images[frame][4].find("TIC")
            tic_and_next_five = images[frame][4][tic_index:tic_index + 12]
            object_text1.set_text(f'Object: {tic_and_next_five}')
            object_text2.set_text(f'Object: {tic_and_next_five}')

        else:
            object_text1.set_text(f'Object: {images[frame][4][:11]}')
            object_text2.set_text(f'Object: {images[frame][4][:11]}')

        return [im1, im2, time_text1, object_text1, frame_text1, info_text1, time_text2, object_text2, frame_text2,
                info_text2]

    animation = FuncAnimation(fig, update, frames=len(images), blit=True)
    animation.save(output_path, writer='imagemagick', fps=5)
    print(f"Animation saved to: {output_path}")


def find_current_night_directory(file_path):
    # Get the current date in the format YYYYMMDD
    current_date = datetime.now().strftime("%Y%m%d") + '/'
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d") + '/'

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(file_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return None


def process_images_by_prefix(base_path, save_path):
    # Find the current night directory
    current_night_directory = find_current_night_directory(base_path)

    if current_night_directory:
        print(f"Current night directory found: {current_night_directory}")

        # Get a list of all fits files in the current night directory
        fits_files = find_fits(current_night_directory)

        # Group fits files by their prefixes
        prefix_groups = defaultdict(list)
        for fits_file in fits_files:
            # Extract the file name (excluding the directory path)
            file_name = os.path.basename(fits_file)
            prefix = file_name[:11]
            prefix_groups[prefix].append(fits_file)

        for prefix, fits_files_with_prefix in prefix_groups.items():
            print(f"\nProcessing images with prefix: {prefix}")
            images = read_fits(fits_files_with_prefix)

            # Check if the images list is not empty before proceeding
            if images:
                registered_images = register_images(images)
                print(f"Number of registered_images used: {len(registered_images)}")

                # Create and save animation for each prefix
                create_blink_animation(registered_images, save_path)
            else:
                print(f"No fits images found for prefix: {prefix}")

    else:
        print("No current night directory found.")


def main():
    plot_images()
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

    # Process images for each prefix in the current night directory
    process_images_by_prefix(base_path, save_path)


if __name__ == "__main__":
    main()
