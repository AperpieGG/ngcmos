import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from utils import plot_images

plot_images()

# Load the CSV file with headers on the second line
file_path = '/Users/u5500483/Downloads/goto_filters.csv'  # Replace with the actual file path
data = pd.read_csv(file_path, header=1)
print(data.head())

qe_ccd = data['QE_CCD']
wv_ccd = data['WV_CCD']
ngts_tra = data['NGTS_TRA']
wv_ngts = data['WV_NGTS']
tel_tra = data['TEL_TRA']
wv_tel = data['WV_TEL']


def smoothen_curve(x, y, smooth):

    spline = UnivariateSpline(x, y, s=smooth)

    smooth_x = np.linspace(x.min(), x.max(), 1000)
    smooth_y = spline(smooth_x)

    return smooth_x, smooth_y


# Extract wavelength and %T for L_band and sloan_g
wavelength_l_band = data['Wavelength (nm)']  # Assuming column name for L_band wavelength
l_band_transmission = data['%T_L']

wavelength_sloan_g = data['Wavelength (nm)']  # Assuming column name for sloan_g wavelength
sloan_g_transmission = data['%T_g']

wavelength_sloan_i = data['Wavelength (nm)']  # Assuming column name for sloan_i wavelength
sloan_i_transmission = data['%T_i']

wavelength_sloan_r = data['Wavelength (nm)']  # Assuming column name for sloan_r wavelength
sloan_r_transmission = data['%T_r']

wavelength_bessel_i = data['Wavelength (nm)']  # Assuming column name for bessel wavelength
bessel_i_transmission = data['%T_bi']


# Ensure no NaN values in wv_cmos and qe_cmos
data = data.dropna(subset=['WV_CMOS', 'QE_CMOS'])

# Sort wv_cmos and qe_cmos in ascending order
data = data.sort_values(by='WV_CMOS')

# Extract sorted wv_cmos and qe_cmos
wv_cmos = data['WV_CMOS']
qe_cmos = data['QE_CMOS']

# Smooth the curve after sorting
wv_cmos, qe_cmos = smoothen_curve(wv_cmos, qe_cmos, 400)

# Plot the data
plt.figure()
plt.plot(wavelength_l_band, l_band_transmission, label="L_band %T", color="purple")
# plt.plot(wavelength_sloan_g, sloan_g_transmission, label="sloan_g %T", color="navy")
# plt.plot(wavelength_sloan_i,  sloan_i_transmission, label="sloan_i %T", color="blue")
# plt.plot(wavelength_sloan_r, sloan_r_transmission, label="sloan_r %T", color="red")
# plt.plot(wavelength_bessel_i, bessel_i_transmission, label="bessel_i %T", color="darkred")

plt.plot(wv_cmos, qe_cmos, label="QE_CMOS", color="red")
plt.plot(wv_ccd, qe_ccd, label="QE_CCD", color="blue")
plt.plot(wv_ngts, ngts_tra, label="NGTS_TRA", color="orange")
plt.plot(wv_tel, tel_tra, label="TEL_TRA", color="green")

# Add labels and title
plt.xlabel("Wavelength (nm)")
plt.ylabel("% Transmission / Quantum Efficiency")
plt.ylim(0, 100)
plt.xlim(379, 1000)
# plt.legend()
plt.grid(True)

# Show the plot
plt.show()