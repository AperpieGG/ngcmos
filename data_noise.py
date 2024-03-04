#! /usr/bin/env python
import numpy as np
import os
from noise import plot_images, noise_model
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

    # Iterate through each photometry file
    for phot_file in phot_files:
        print(f"Plotting photometry file {phot_file}...")
        phot_table = read_phot_file(phot_file)

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
            RMS = np.std(dt_flux_binned)
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
            read_signal = (read_noise_pix * npix) ** 2

            # Call the noise model function with real data
            noise_model(flux, photon_shot_noise, sky, sky_noise, read_noise, read_signal, dark_current, dc_noise)

            # Plot or save the noise model plot here if needed


if __name__ == "__main__":
    main()



