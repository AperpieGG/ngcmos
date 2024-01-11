import datetime
import glob
import numpy as np
from astropy.io import fits
from astropy.visualization import SqrtStretch, ImageNormalize, ZScaleInterval, AsinhStretch
from image_registration import chi2_shift
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_fits_image(file_path):
    with fits.open(file_path) as hdu_list:
        image_data = hdu_list[0].data
        image_data = image_data[450:550, 600:700]
    return image_data


def register_images(images):
    # Assuming the first image as the reference
    reference_image = images[0]
    reference_data = reference_image.data

    # Register each image to the reference using chi2_shift
    registered_images_list = [reference_image]
    for i in range(1, len(images), 5):
        img = images[i]
        # Ensure img is a NumPy array
        img_data = img.data

        # Use chi2_shift without the WCS information
        shift_result = chi2_shift(reference_data, img_data)

        # Shift the data using the returned values
        shifted_image_data = np.roll(img_data, shift_result[0].astype(int), axis=0)

        # Create a new HDU with the shifted data and the original header
        shifted_hdu = fits.PrimaryHDU(data=shifted_image_data)
        registered_images_list.append(shifted_hdu)

    return registered_images_list


def create_blink_animation(images, output_path=datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.gif'):
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up the normalization for the images
    zscale_interval = ZScaleInterval()
    norm = ImageNormalize(interval=zscale_interval, stretch=SqrtStretch())

    # Initialize the plot with the first image
    im = ax.imshow(images[0].data, cmap='hot', origin='lower', norm=norm)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('QC guiding')

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                        fontsize=10, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.8))

    def update(frame):
        im.set_array(images[frame].data)
        time_text.set_text('Frame: {}'.format(frame))
        return [im, time_text]

    animation = FuncAnimation(fig, update, frames=len(images), blit=True)
    animation.save(output_path, writer='imagemagick', fps=5)


if __name__ == "__main__":
    # Replace these paths with your actual file paths
    directory_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/TOI-00451/'

    # Read the FITS files
    fits_files = sorted(glob.glob(directory_path + '*_r.fits'))

    # Read and register the images
    raw_images = [read_fits_image(file) for file in fits_files]
    registered_images = register_images(raw_images)

    # Create and save the blink animation
    create_blink_animation(registered_images)
