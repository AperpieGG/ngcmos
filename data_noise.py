#! /usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
import json
from astropy.io import fits
from datetime import datetime, timedelta
import fnmatch


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    If not found, use the current working directory.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    return current_date_directory if os.path.isdir(current_date_directory) else os.getcwd()


def get_phot_files(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'phot_*.fits'):
            phot_files.append(filename)
    return phot_files


def read_phot_file(filename):
    """
    Read the photometry file.
    """
    try:
        with fits.open(filename) as ff:
            tab = ff[1].data
            return tab
    except Exception as e:
        print(f"Error reading photometry file {filename}: {e}")
        return None


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
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14


def bin_time_flux_error(time, flux, error, bin_fact):
    n_binned = int(len(time) / bin_fact)
    clip = n_binned * bin_fact
    time_b = np.average(time[:clip].reshape(n_binned, bin_fact), axis=1)
    if len(flux.shape) == 1:
        flux_b = np.average(flux[:clip].reshape(n_binned, bin_fact), axis=1)
        error_b = np.sqrt(np.sum(error[:clip].reshape(n_binned, bin_fact) ** 2, axis=1)) / bin_fact
    else:
        n_stars = len(flux)
        flux_b = np.average(flux[:clip].reshape((n_stars, n_binned, bin_fact)), axis=2)
        error_b = np.sqrt(np.sum(error[:clip].reshape((n_stars, n_binned, bin_fact)) ** 2, axis=2)) / bin_fact
    return time_b, flux_b, error_b


def scintillation_noise():
    t = 10  # exposure time
    D = 20  # telescope diameter
    secZ = 1.2  # airmass
    h = 2433  # height of Paranal
    ho = 8000  # height of atmospheric scale
    W = 1.75  # wind speed
    N = 0.09 * (D ** (-2 / 3) * secZ ** W * np.exp(-h / ho)) * (2 * t) ** (-1 / 2)
    return N


def noise_model(flux, photon_shot_noise, sky_noise, read_noise, dc_noise, N):
    N_sc = (N * flux) ** 2
    total_noise = np.sqrt(photon_shot_noise + read_noise + dc_noise + N_sc + sky_noise ** 2)
    photon_shot_noise /= flux
    sky_noise /= flux
    read_noise /= flux
    dc_noise /= flux
    N_sc /= flux
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(flux, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(flux, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(flux, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(flux, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(flux, N, color='orange', label='scintillation noise', linestyle='--')
    ax.plot(flux, total_noise, color='black', label='total noise')
    ax.set_xlabel('Flux')
    ax.set_ylabel('RMS (mag)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


def main():
    plot_images()
    current_night_directory = find_current_night_directory(base_path)
    phot_files = get_phot_files(current_night_directory)
    num_stars = 50
    bin_size = 60

    all_flux = []
    all_sky = []
    all_photon_shot_noise = []
    all_sky_noise = []
    all_dc_noise = []
    all_read_noise = []
    all_N = []

    if phot_files:
        for phot_file in phot_files:
            phot_table = read_phot_file(phot_file)
            if phot_table is not None:
                for gaia_id in phot_table['gaia_id'][:num_stars]:
                    gaia_id_data = phot_table[phot_table['gaia_id'] == gaia_id]
                    jd_mid = gaia_id_data['jd_mid']
                    flux_2 = gaia_id_data['flux_2']
                    fluxerr_2 = gaia_id_data['fluxerr_2']
                    flux_w_sky_2 = gaia_id_data['flux_w_sky_2']
                    sky_2 = flux_w_sky_2 - flux_2

                    trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), flux_2, 2), jd_mid - int(jd_mid[0]))
                    dt_flux = flux_2 / trend
                    dt_fluxerr = fluxerr_2 / trend

                    time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, dt_flux, dt_fluxerr,
                                                                                         bin_size)

                    mean_flux = np.mean(flux_2)
                    mean_sky = np.mean(sky_2)

                    flux = mean_flux
                    sky = mean_sky
                    photon_shot_noise = np.sqrt(flux) / flux
                    sky_noise = np.sqrt(sky) / flux

                    aperture_radius = 3
                    npix = np.pi * aperture_radius ** 2

                    exposure_time = 10
                    dark_current_rate = 0.66
                    dark_current = dark_current_rate * exposure_time * npix
                    dc_noise = np.sqrt(dark_current) / flux

                    read_noise_pix = 1.56
                    read_noise = (read_noise_pix * npix) / flux

                    N = scintillation_noise()

                    all_flux.append(flux)
                    all_sky.append(sky)
                    all_photon_shot_noise.append(photon_shot_noise)
                    all_sky_noise.append(sky_noise)
                    all_dc_noise.append(dc_noise)
                    all_read_noise.append(read_noise)
                    all_N.append(N)

    all_flux = np.array(all_flux)
    all_sky = np.array(all_sky)
    all_photon_shot_noise = np.array(all_photon_shot_noise)
    all_sky_noise = np.array(all_sky_noise)
    all_dc_noise = np.array(all_dc_noise)
    all_read_noise = np.array(all_read_noise)
    all_N = np.array(all_N)

    noise_model(all_flux, all_photon_shot_noise, all_sky_noise, all_read_noise, all_dc_noise, all_N)


if __name__ == "__main__":
    main()
