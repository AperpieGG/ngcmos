import numpy as np
import matplotlib.pyplot as plt
import math

# pylint: disable=superfluous-parens
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name


def plot_images():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
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

    # Font and fontsize

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # Legend

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14


def calculate_camera_spec():

    # Camera specifications
    pixel_size_mm = 11e-3  # 11 microns converted to millimeters
    sensor_resolution = (2048, 2048)

    # Telescope specifications
    telescope_focal_length_mm = 560  # NGTS telescope focal length in mm

    # Calculate horizontal and vertical FOV in degrees
    fov_x_deg = math.degrees(2 * math.atan((sensor_resolution[0] * pixel_size_mm) / (2 * telescope_focal_length_mm)))
    fov_y_deg = math.degrees(2 * math.atan((sensor_resolution[1] * pixel_size_mm) / (2 * telescope_focal_length_mm)))

    # Calculate diagonal FOV using Pythagorean theorem
    fov_diagonal_deg = math.sqrt(fov_x_deg ** 2 + fov_y_deg ** 2)

    # Calculate pixel scale in arcseconds
    pixel_scale_arcsec = 11 * 206.265 / telescope_focal_length_mm

    print(f"Pixel Scale: {pixel_scale_arcsec:.2f} arcseconds per pixel")

    print(f"Diagonal FOV: {fov_diagonal_deg:.2f} degrees")

    print(f"FOV in arcminutes: {fov_diagonal_deg * 60:.2f} arcminutes")


def noise_sources():
    """
    Returns the noise sources for a given flux

    returns arrays of noise and signal for a given flux

    Parameters
    ----------
    None

    Returns
    -------
    flux : array
        The flux of the star in electrons per second
    photon_shot_noise : array
        The photon shot noise
    sky_flux : array
        The sky flux
    sky_noise : array
        The sky noise
    read_noise : array
        The read noise
    read_signal : array
        The read signal
    dark_current : array
        The dark current
    dc_noise : array
        The dark current noise

    """

    aperture_radius = 3
    npix = np.pi * aperture_radius ** 2

    # set exposure time and and random flux
    exposure_time = 10
    flux = np.arange(100, 1e6, 10)

    # set dark current rate from cmos characterisation
    dark_current_rate = 0.66
    dark_current = dark_current_rate * exposure_time * npix
    dc_noise = np.sqrt(dark_current) / flux

    # set read noise from cmos characterisation
    read_noise_pix = 1.74
    read_noise = (read_noise_pix * npix) / flux
    read_signal = (read_noise_pix * npix) ** 2

    # set random sky background
    sky_flux = 22.6 * exposure_time * npix
    sky_noise = np.sqrt(sky_flux) / flux

    # set random photon shot noise from the flux
    photon_shot_noise = np.sqrt(flux) / flux

    return flux, photon_shot_noise, sky_flux, sky_noise, read_noise, read_signal, dark_current, dc_noise


def scintilation_noise():
    t = 10  # exposure time
    D = 20  # telescope diameter
    secZ = 1.2  # airmass
    h = 2433  # height of Paranal
    ho = 8000  # height of atmospheric scale
    W = 1.75  # wind speed
    N = 0.09 * (D ** (-2 / 3) * secZ ** W * np.exp(-h / ho)) * (2 * t) ** (-1 / 2)
    return N


def noise_model(flux, photon_shot_noise, sky_flux, sky_noise, read_noise, read_signal, dark_current, dc_noise):
    N = scintilation_noise()
    N_sc = (N * flux) ** 2

    total_noise = np.sqrt(flux + sky_flux + dark_current + read_signal + N_sc)
    RNS = total_noise / flux
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(flux, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(flux, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(flux, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(flux, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(flux, np.ones(len(flux)) * N, color='orange', label='scintilation noise', linestyle='--')
    ax.plot(flux, RNS, color='black', label='total noise')
    ax.set_xlabel('Flux')
    ax.set_ylabel('RMS (mag)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.001, 0.1)
    ax.set_xlim(1000, 1e6)
    plt.tight_layout()

    plt.legend(loc='best')
    plt.show()


plot_images()
flux, photon_shot_noise, sky_flux, sky_noise, read_noise, read_signal, dark_current, dc_noise = noise_sources()
noise_model(flux, photon_shot_noise, sky_flux, sky_noise, read_noise, read_signal, dark_current, dc_noise)

calculate_camera_spec()

# TODO add noise model for the Ikon-L

