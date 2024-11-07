from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from utils import plot_images

plot_images()


def load_fits_image(file_path):
    """Load a FITS image and return the data array."""
    with fits.open(file_path) as hdul:
        data = hdul[0].data
    return data


def calculate_vignetting_pattern(image):
    """Estimate vignetting by calculating radial brightness fall-off."""
    # Define the centre of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

    # Calculate the distance of each pixel from the centre
    y, x = np.indices(image.shape)
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Calculate radial profile by averaging values at each radius
    max_radius = int(np.max(distance_from_center))
    radial_profile = np.zeros(max_radius)
    counts = np.zeros(max_radius)

    for r in range(max_radius):
        mask = (distance_from_center >= r) & (distance_from_center < r + 1)
        radial_profile[r] = np.mean(image[mask])
        counts[r] = np.sum(mask)

    # Normalise radial profile by central brightness
    radial_profile /= radial_profile[0]
    return radial_profile


def perform_statistics(radial_profile):
    """Calculate mean and standard deviation of vignetting profile."""
    mean_vignetting = np.mean(radial_profile)
    std_vignetting = np.std(radial_profile)
    return mean_vignetting, std_vignetting


# Paths to your FITS files
file_path1 = 'master_flat_L.fits'
file_path2 = 'master_flat_NGTS.fits'
file_path3 = 'master_flat_g.fits'

# Load FITS files
image1 = load_fits_image(file_path1)
image2 = load_fits_image(file_path2)
image3 = load_fits_image(file_path3)

# Calculate vignetting patterns
vignetting_profile1 = calculate_vignetting_pattern(image1)
vignetting_profile2 = calculate_vignetting_pattern(image2)
vignetting_profile3 = calculate_vignetting_pattern(image3)

# Perform statistics on vignetting profiles
mean_vignetting1, std_vignetting1 = perform_statistics(vignetting_profile1)
mean_vignetting2, std_vignetting2 = perform_statistics(vignetting_profile2)
mean_vignetting3, std_vignetting3 = perform_statistics(vignetting_profile3)

# Print statistics
print(f"Filter L - Mean Vignetting: {mean_vignetting1:.3f}, Std Dev: {std_vignetting1:.3f}")
print(f"Filter NGTS - Mean Vignetting: {mean_vignetting2:.3f}, Std Dev: {std_vignetting2:.3f}")
print(f"Filter g - Mean Vignetting: {mean_vignetting3:.3f}, Std Dev: {std_vignetting3:.3f}")

# Plot vignetting profiles for comparison
plt.plot(vignetting_profile1, label='Filter L', color='purple')
plt.plot(vignetting_profile2, label='Filter NGTS', color='orange')
plt.plot(vignetting_profile3, label='Filter g', color='blue')
plt.xlabel('Distance from Centre (pixels)')
plt.ylabel('Normalised Brightness')
plt.title('Vignetting Profiles for Three Filters')
plt.legend()
plt.show()