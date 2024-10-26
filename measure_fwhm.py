import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from astropy.time import Time
import astropy.units as u
from utils import get_location, get_light_travel_times
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


# Function to fit a 2D Gaussian
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


# Get airmass from altitude
def calculate_airmass(altitude):
    return 1 / np.cos(np.radians(90 - altitude))


# Function to calculate FWHM for stars in an image
def calculate_fwhm(image_data, crop_size=800):
    center_x, center_y = image_data.shape[1] // 2, image_data.shape[0] // 2
    cropped_image_data = image_data[center_y - crop_size:center_y + crop_size,
                         center_x - crop_size:center_x + crop_size]

    mean, median, std = np.mean(cropped_image_data), np.median(cropped_image_data), mad_std(cropped_image_data)
    daofind = DAOStarFinder(fwhm=4, threshold=5. * std, brightest=50)
    selected_sources = daofind(cropped_image_data - median)

    fwhms_x, fwhms_y = [], []
    for x_star, y_star in zip(selected_sources['xcentroid'], selected_sources['ycentroid']):
        x_start, x_end = int(x_star) - 3, int(x_star) + 3
        y_start, y_end = int(y_star) - 3, int(y_star) + 3

        if x_end > x_start and y_end > y_start:
            x = np.arange(x_end - x_start)
            y = np.arange(y_end - y_start)
            x, y = np.meshgrid(x, y)
            sub_image = cropped_image_data[y_start:y_end, x_start:x_end]
            initial_guess = (np.max(sub_image), (x_end - x_start) // 2, (y_end - y_start) // 2, 3, 3, 0, 0)

            try:
                popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), sub_image.ravel(), p0=initial_guess)
                sigma_x, sigma_y = popt[3], popt[4]
                fwhms_x.append(2.355 * sigma_x)
                fwhms_y.append(2.355 * sigma_y)
            except:
                continue

    if fwhms_x and fwhms_y:
        return np.median(fwhms_x + fwhms_y) / 2
    return None


# Process each FITS file in the directory
directory = os.getcwd()
times, fwhm_values, airmass_values = [], [], []

for filename in os.listdir(directory):
    if filename.endswith('.fits'):
        with fits.open(filename, mode='update') as hdul:
            header = hdul[0].header
            image_data = hdul[0].data

            # Get BJD from header or calculate if missing
            if 'BJD' not in header:
                exptime = float(header['EXPTIME'])
                time_isot = Time(header['DATE-OBS'], format='isot', scale='utc', location=get_location())
                time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
                time_jd += (exptime / 2.) * u.second
                ra, dec = header['TELRAD'], header['TELDECD']
                ltt_bary, _ = get_light_travel_times(ra, dec, time_jd)
                time_bary = time_jd.tdb + ltt_bary
                header['BJD'] = time_bary.value

            # Get airmass from header or calculate if missing
            if 'AIRMASS' not in header:
                altitude = header.get('ALTITUDE', 45)  # Example: default to 45 if missing
                header['AIRMASS'] = calculate_airmass(altitude)

            # Append BJD, FWHM, and Airmass
            times.append(header['BJD'])
            fwhm = calculate_fwhm(image_data)
            if fwhm:
                fwhm_values.append(fwhm)
                airmass_values.append(header['AIRMASS'])

# Sort by BJD for plotting
sorted_data = sorted(zip(times, fwhm_values, airmass_values), key=lambda x: x[0])
times, fwhm_values, airmass_values = zip(*sorted_data)

# Plot FWHM vs Time and FWHM vs Airmass
fig, ax1 = plt.subplots()

ax1.plot(times, fwhm_values, 'o', label="FWHM vs. BJD", color="blue")
ax1.set_xlabel("BJD")
ax1.set_ylabel("FWHM (pixels)", color="blue")

# Airmass on top x-axis
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Airmass')
interpolated_airmass = np.interp(ax1.get_xticks(), times, airmass_values)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')

ax1.legend()
plt.tight_layout()
plt.show()
