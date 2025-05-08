#! /usr/bin/env python
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_images, read_phot_file, get_phot_files
from astropy.io import fits
import matplotlib.patches as patches

plot_images()

good_cmos_stars = [169763812, 188619865, 188620343, 188620477, 188622268, 188627904, 188628755, 214657985, 214661799,
                   214661930, 270187208]

bad_cmos_stars = [169763929, 4611043, 5796255, 5796320, 5796376, 169746092, 169746369, 169746459, 169763609, 169763615,
                  169763631, 169763985, 169764011, 169764168, 169764174, 188620052, 188620450, 188620644, 188622237,
                  188622275, 188622523, 188628115, 188628237, 188628252, 188628309, 188628413, 188628448, 188628555,
                  188628748, 214657492, 214658021, 214661588, 214662807, 214662895, 214662905, 214664699, 214664842,
                  270185125, 270185254, 270187139, 270187283]


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


def main():
    # Define image directory
    current_night_directory = '.'

    # Get photometry files
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Read first photometry file
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_files[0]))

    # Use the first frame ID for the image
    first_frame_id = phot_table['frame_id'][0]
    image_data = get_image_data(first_frame_id, image_directory=current_night_directory)
    if image_data is None:
        print(f"Image data for frame_id {first_frame_id} not found.")
        return

    # Start plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=np.percentile(image_data, 5),
              vmax=np.percentile(image_data, 99))

    # Plot good and bad stars with colored circles
    for tic_id in good_cmos_stars + bad_cmos_stars:
        match = phot_table[phot_table['tic_id'] == tic_id]
        if len(match) == 0:
            print(f"TIC ID {tic_id} not found in the photometry file.")
            continue

        x, y = match['x'][0], match['y'][0]
        color = 'lime' if tic_id in good_cmos_stars else 'red'
        circ = patches.Circle((x, y), radius=10, edgecolor=color, facecolor='none', linewidth=1.5)
        ax.add_patch(circ)

    ax.set_title(f'Frame: {first_frame_id}')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    main()
