#! /usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
from analyse_noise import load_config, find_current_night_directory, get_phot_files, read_phot_file

# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


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

    plt.rcParams['font.size'] = 14

    # Legend

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14


def bin_time_flux_error(time, flux, error, bin_fact):
    """
    Use reshape to bin light curve data, clip under filled bins
    Works with 2D arrays of flux and errors

    Note: under filled bins are clipped off the end of the series

    Parameters
    ----------
    time : array         of times to bin
    flux : array         of flux values to bin
    error : array         of error values to bin
    bin_fact : int
        Number of measurements to combine

    Returns
    -------
    times_b : array
        Binned times
    flux_b : array
        Binned fluxes
    error_b : array
        Binned errors

    Raises
    ------
    None
    """
    n_binned = int(len(time) / bin_fact)
    clip = n_binned * bin_fact
    time_b = np.average(time[:clip].reshape(n_binned, bin_fact), axis=1)
    # determine if 1 or 2d flux/err inputs
    if len(flux.shape) == 1:
        flux_b = np.average(flux[:clip].reshape(n_binned, bin_fact), axis=1)
        error_b = np.sqrt(np.sum(error[:clip].reshape(n_binned, bin_fact) ** 2, axis=1)) / bin_fact
    else:
        # assumed 2d with 1 row per star
        n_stars = len(flux)
        flux_b = np.average(flux[:clip].reshape((n_stars, n_binned, bin_fact)), axis=2)
        error_b = np.sqrt(np.sum(error[:clip].reshape((n_stars, n_binned, bin_fact)) ** 2, axis=2)) / bin_fact
    return time_b, flux_b, error_b


def calculate_mean_rms_binned(table, bin_size, num_stars):
    mean_flux_list = []
    RMS_list = []
    mean_sky_list = []

    for gaia_id in table['gaia_id'][:num_stars]:  # Selecting the first num_stars stars
        gaia_id_data = table[table['gaia_id'] == gaia_id]
        jd_mid = gaia_id_data['jd_mid']
        flux_2 = gaia_id_data['flux_2']
        fluxerr_2 = gaia_id_data['fluxerr_2']
        flux_w_sky_2 = gaia_id_data['flux_w_sky_2']
        sky_2 = flux_w_sky_2 - flux_2

        trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), flux_2, 2), jd_mid - int(jd_mid[0]))
        dt_flux = flux_2 / trend
        dt_fluxerr = fluxerr_2 / trend

        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, dt_flux, dt_fluxerr, bin_size)

        # Calculate mean flux, sky and RMS
        mean_flux = np.mean(flux_2)
        RMS = np.std(dt_flux_binned)
        mean_sky = np.mean(sky_2)

        # Append to lists
        mean_flux_list.append(mean_flux)
        RMS_list.append(RMS)
        mean_sky_list.append(mean_sky)

    print(f"The length of the RMS list is {len(RMS_list)}")
    return mean_flux_list, RMS_list, mean_sky_list


def scintilation_noise():
    t = 10  # exposure time
    D = 20  # telescope diameter
    secZ = 1.2  # airmass
    h = 2433  # height of Paranal
    ho = 8000  # height of atmospheric scale
    W = 1.75  # wind speed
    N = 0.09 * (D ** (-2 / 3) * secZ ** W * np.exp(-h / ho)) * (2 * t) ** (-1 / 2)
    return N


def noise_model(flux, photon_shot_noise, sky_flux, sky_noise, read_noise, dc_noise, N):
    # Calculate scintillation noise contribution
    N_sc = (N * flux) ** 2

    # Calculate total noise
    total_noise = np.sqrt(photon_shot_noise + read_noise + dc_noise + N_sc + sky_noise ** 2)

    # Normalize all noise components with respect to flux
    photon_shot_noise /= flux
    sky_noise /= flux
    read_noise /= flux
    dc_noise /= flux
    N_sc /= flux

    # Plot the noise model
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
    ax.set_ylim(0.001, 0.1)
    ax.set_xlim(1000, 1e6)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


def main():
    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Set the number of stars to process
    num_stars = 50
    # Set the bin size
    bin_size = 60

    # Initialize lists to store aggregated data
    all_flux = []
    all_sky = []
    all_photon_shot_noise = []
    all_sky_noise = []
    all_dc_noise = []
    all_read_noise = []
    all_N = []

    # Iterate through each photometry file
    for phot_file in phot_files:
        print(f"Processing photometry file {phot_file}...")
        phot_table = read_phot_file(phot_file[0])

        # Calculate mean and RMS for the noise model for each star
        for gaia_id in phot_table['gaia_id'][:num_stars]:
            gaia_id_data = phot_table[phot_table['gaia_id'] == gaia_id]
            jd_mid = gaia_id_data['jd_mid']
            flux_2 = gaia_id_data['flux_2']
            fluxerr_2 = gaia_id_data['fluxerr_2']
            flux_w_sky_2 = gaia_id_data['flux_w_sky_2']
            sky_2 = flux_w_sky_2 - flux_2

            # Detrend the flux
            trend = np.polyval(np.polyfit(jd_mid - int(jd_mid[0]), flux_2, 2), jd_mid - int(jd_mid[0]))
            dt_flux = flux_2 / trend
            dt_fluxerr = fluxerr_2 / trend

            # Bin the data if needed
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, dt_flux, dt_fluxerr, bin_size)

            # Calculate mean flux, sky, and RMS
            mean_flux = np.mean(flux_2)
            mean_sky = np.mean(sky_2)

            # Pass real data to the noise model
            flux = mean_flux
            sky = mean_sky
            photon_shot_noise = np.sqrt(flux) / flux
            sky_noise = np.sqrt(sky) / flux

            aperture_radius = 3
            npix = np.pi * aperture_radius ** 2

            # Set exposure time
            exposure_time = 10

            # Set dark current rate from CMOS characterization
            dark_current_rate = 0.66
            dark_current = dark_current_rate * exposure_time * npix
            dc_noise = np.sqrt(dark_current) / flux

            # Set read noise from CMOS characterization
            read_noise_pix = 1.56
            read_noise = (read_noise_pix * npix) / flux

            # Calculate scintillation noise
            N = scintilation_noise()

            # Append data to aggregated lists
            all_flux.append(flux)
            all_sky.append(sky)
            all_photon_shot_noise.append(photon_shot_noise)
            all_sky_noise.append(sky_noise)
            all_dc_noise.append(dc_noise)
            all_read_noise.append(read_noise)
            all_N.append(N)

    # Convert lists to numpy arrays
    all_flux = np.array(all_flux)
    all_sky = np.array(all_sky)
    all_photon_shot_noise = np.array(all_photon_shot_noise)
    all_sky_noise = np.array(all_sky_noise)
    all_dc_noise = np.array(all_dc_noise)
    all_read_noise = np.array(all_read_noise)
    all_N = np.array(all_N)

    # Call the noise model function with aggregated data
    noise_model(all_flux, all_photon_shot_noise, all_sky, all_sky_noise, all_read_noise, all_dc_noise, all_N)


if __name__ == "__main__":
    main()


