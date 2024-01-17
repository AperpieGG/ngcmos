#!/Users/u5500483/anaconda3/bin/python
from datetime import datetime, timedelta
import glob
import os
import numpy as np
from astropy.io import fits
from astropy.visualization import SqrtStretch, ImageNormalize, ZScaleInterval
from image_registration import chi2_shift
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
    raw_images = filtered_files[::5]
    return raw_images


def read_fits(file_path):
    images_dt = []
    for file_path in file_path:
        with fits.open(file_path) as hdu_list:
            image_data = hdu_list[0].data[450:550, 600:700]
            date_obs = hdu_list[0].header.get('DATE-OBS', 'N/A')
            tel_ra = hdu_list[0].header.get('TELRA', 'N/A')
            tel_dec = hdu_list[0].header.get('TELDEC', 'N/A')
            target_id = hdu_list[0].header.get('OBJECT', 'N/A')
            images_dt.append((image_data, date_obs, tel_ra, tel_dec, target_id))
    return images_dt


def register_images(images):
    reference_image, reference_date, reference_ra, reference_dec, obj = images[0]
    reference_data = reference_image

    registered_images_list = [(reference_data, reference_date, reference_ra, reference_dec, obj)]
    for img, date_obs, tel_ra, tel_dec, obj in images[1:]:
        shift_result = chi2_shift(reference_data, img)
        shifted_image_data = np.roll(img, shift_result[0].astype(int), axis=0)
        registered_images_list.append((shifted_image_data, date_obs, tel_ra, tel_dec, obj))

    return registered_images_list


def create_blink_animation(images, save_path):
    # Default output path with object name and date
    object_name = images[0][4][:11]
    timestamp = datetime.now().strftime('%Y%m%d')
    output_name = f"donuts_{object_name}_{timestamp}.gif"
    output_path = os.path.join(save_path, output_name)

    # Create the base path directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    zscale_interval = ZScaleInterval()
    norm = ImageNormalize(interval=zscale_interval, stretch=SqrtStretch())
    im = ax.imshow(images[0][0], cmap='hot', origin='lower', norm=norm)
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

    if "TOI" in images[0][4]:
        object_text = ax.text(0.78, 0.02, '', transform=ax.transAxes, color='white',
                              fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.8))
    elif "TIC" in images[0][4]:
        object_text = ax.text(0.74, 0.02, '', transform=ax.transAxes, color='white',
                              fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.8))
    else:
        pass

    def update(frame):
        im.set_array(images[frame][0])
        time_text.set_text(f'DATE-OBS: {images[frame][1]}')
        frame_text.set_text(f'Frame: {frame + 1}')
        object_ra_dec_text = f'RA: {images[frame][2]}, DEC: {images[frame][3]}'
        info_text.set_text(object_ra_dec_text)

        object_text.set_text(f'Object: {images[frame][4][:11]}')
        if "TOI" in images[frame][4]:
            # Find the index where "TOI" starts
            toi_index = images[frame][4].find("TOI")

            # Extract "TOI" and the next 5 letters
            toi_and_next_five = images[frame][4][toi_index:toi_index + 9]

            object_text.set_text(f'Object: {toi_and_next_five}')
        elif "TIC" in images[frame][4]:
            # Find the index where "TIC" starts
            tic_index = images[frame][4].find("TIC")

            # Extract "TIC" and the next 5 letters
            tic_and_next_five = images[frame][4][tic_index:tic_index + 15]

            object_text.set_text(f'Object: {tic_and_next_five}')
        else:
            object_text.set_text(f'Object: {images[frame][4][:11]}')

        return [im, time_text, object_text, frame_text, info_text]

    animation = FuncAnimation(fig, update, frames=len(images), blit=True)
    animation.save(output_path, writer='imagemagick', fps=5)


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


if __name__ == "__main__":
    base_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/'
    save_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/shifts_plots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Find the current night directory
    current_night_directory = find_current_night_directory(base_path)
    plot_images()

    if current_night_directory:
        print(f"Current night directory found: {current_night_directory}")
        images = read_fits(find_fits(current_night_directory))

        # Check if the images list is not empty before proceeding
        if images:
            registered_images = register_images(images)
            print(f"Total number of registered_images used: {len(registered_images)}")
            create_blink_animation(registered_images, save_path)
        else:
            print("No fits images found.")
    else:
        print("No current night directory found.")
