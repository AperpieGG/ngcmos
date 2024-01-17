#!/Users/u5500483/anaconda3/bin/python
import datetime
import glob
import os
import numpy as np
from astropy.io import fits
from astropy.visualization import SqrtStretch, ImageNormalize, ZScaleInterval
from image_registration import chi2_shift
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_fits_image(file_path):
    with fits.open(file_path) as hdu_list:
        image_data = hdu_list[0].data[450:550, 600:700]
        date_obs = hdu_list[0].header.get('DATE-OBS', 'N/A')
        tel_ra = hdu_list[0].header.get('TELRA', 'N/A')
        tel_dec = hdu_list[0].header.get('TELDEC', 'N/A')
        target_id = hdu_list[0].header.get('OBJECT', 'N/A')
    return image_data, date_obs, tel_ra, tel_dec, target_id


def register_images(images):
    reference_image, reference_date, reference_ra, reference_dec, obj = images[0]
    reference_data = reference_image

    registered_images_list = [(reference_data, reference_date, reference_ra, reference_dec, obj)]
    for img, date_obs, tel_ra, tel_dec, obj in images[1:]:
        shift_result = chi2_shift(reference_data, img)
        shifted_image_data = np.roll(img, shift_result[0].astype(int), axis=0)
        registered_images_list.append((shifted_image_data, date_obs, tel_ra, tel_dec, obj))

    return registered_images_list


def create_blink_animation(images, output_path=None):
    if save_path is not None:
        # Default output path with object name and date
        object_name = images[0][4][:11]
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
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


if __name__ == "__main__":
    directory_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/TOI-00451/'
    save_path = "/Users/u5500483/Downloads/DATA_MAC/CMOS/shifts_plots/"
    fits_files = sorted(glob.glob(directory_path + '*_r.fits'))

    raw_images = [read_fits_image(file) for file in fits_files[::5]]
    registered_images = register_images(raw_images)
    print(f"Total number of images used: {len(registered_images)}")

    create_blink_animation(registered_images, save_path)
