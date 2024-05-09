#!/usr/bin/env python

"""
This script calculates the noise model for the TESS data.
The light curves are detrended using a second order polynomial to correct for the airmass.
The noise is calculated from the detrended and normalized light curves.
The fluxes are converted to magnitudes using zero points. The final plot is RMS vs magnitudes.
"""

import argparse
import os
import numpy as np
import json
from utils import (read_phot_file, bin_time_flux_error,
                   remove_outliers, calculate_trend_and_flux, noise_sources, extract_airmass_zp)

APERTURE = 6
READ_NOISE = 1.56
DARK_CURRENT = 1.6


def rms_vs_mags(table, bin_size, num_stars, average_zp):
    """
    Calculate the mean flux and RMS for a given number of stars

    Parameters
    ----------
    table : astropy.table.Table
        Table containing the photometry data
    bin_size : int
        Number of images to bin
    num_stars : int
        Number of stars to plot
    average_zp : float
        Average zero point value

    Returns
    -------
    mean_flux_list : list
        values of mean fluxes
    RMS_list : list
        values of RMS values
    sky_list : list
        values of sky values
    mags_list : list
        values of magnitudes
    Tmags_list : list
        values of Tmag
    """
    mean_flux_list = []
    RMS_list = []
    sky_list = []
    mags_list = []
    Tmags_list = []

    # TODO: instead of phot_files pass the rel_phot_NGFIELD.fits file, the info is Tmag, Flux, RMS, sky

    unique_tic_ids = np.unique(table['tic_id'])
    for tic_id in unique_tic_ids[:num_stars]:  # Selecting the first num_stars unique TIC IDs
        tic_id_data = table[table['tic_id'] == tic_id]
        jd_mid = tic_id_data['jd_mid']
        Tmag = tic_id_data['Tmag'][0]
        flux_4 = tic_id_data['flux_6']
        fluxerr_4 = tic_id_data['fluxerr_6']
        sky_4 = tic_id_data['flux_w_sky_6'] - tic_id_data['flux_6']
        if Tmag > 14:
            continue

        # Apply sigma clipping to flux and sky arrays
        time_clipped, flux_4_clipped, fluxerr_4_clipped = remove_outliers(jd_mid, flux_4, fluxerr_4)

        # Detrend the flux by converting back to fluxes and normalize by the mean lc
        trend, dt_flux, dt_fluxerr = calculate_trend_and_flux(time_clipped, flux_4_clipped, fluxerr_4_clipped)
        # Bin the time, flux, and error
        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(time_clipped, dt_flux, dt_fluxerr,
                                                                             bin_size)

        # Calculate mean flux and RMS
        mean_flux_list.append(np.mean(flux_4_clipped))
        RMS_list.append(np.nanstd(dt_flux_binned) * 1000000)  # Convert to ppm
        sky_list.append(np.median(sky_4))

        # Calculate magnitudes using the average zp
        mags = -2.5 * np.log10(flux_4_clipped / 10) + average_zp
        mags_list.append(np.nanmean(mags))
        Tmags_list.append(round(Tmag, 2))

        print(f"Running for star {tic_id} with Tmag = {Tmag:.2f} and mag = {np.mean(mags):.2f} "
              f"and RMS = {np.std(dt_flux_binned) * 1000000:.2f}")
    print('The max number of stars is: ', len(np.unique(table['tic_id'])))

    return mean_flux_list, RMS_list, sky_list, mags_list, Tmags_list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific tic_id')
    parser.add_argument('filename', type=str, help='Name of the FITS file containing photometry data')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()
    bin_size = args.bin
    filename = args.filename

    # Get the current night directory
    current_night_directory = os.getcwd()

    # Plot the current photometry file
    print(f"Plotting the photometry file {filename}...")
    phot_table = read_phot_file(os.path.join(current_night_directory, filename))

    # Extract airmass and zero point values from the images
    airmass_list, zp = extract_airmass_zp(phot_table, current_night_directory)

    # Get the maximum number of stars based on unique TIC IDs
    max_num_stars = len(np.unique(phot_table['tic_id']))

    # Calculate mean and RMS for the noise model
    mean_flux_list, RMS_list, sky_list, mags_list, Tmags_list = rms_vs_mags(
        phot_table, bin_size=bin_size, num_stars=max_num_stars, average_zp=np.mean(zp))

    # Get noise sources
    synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS = (
        noise_sources(sky_list, bin_size, airmass_list, zp, aper=APERTURE, read_noise=READ_NOISE,
                      dark_current=DARK_CURRENT))

    synthetic_mag_list = synthetic_mag.tolist()
    photon_shot_noise_list = photon_shot_noise.tolist()
    sky_noise_list = sky_noise.tolist()
    read_noise_list = read_noise.tolist()
    dc_noise_list = dc_noise.tolist()
    N_list = N.tolist()
    RNS_list = RNS.tolist()
    Tmags_list = [float(x) for x in Tmags_list]

    # Save RMS_list, mags_list, and other lists to a JSON file
    output_data = {
        "TIC_IDs": np.unique(phot_table['tic_id']).tolist(),  # [int(x) for x in TIC_IDs
        "RMS_list": RMS_list,
        "mags_list": mags_list,
        "Tmag_list": Tmags_list,
        "synthetic_mag": synthetic_mag_list,
        "photon_shot_noise": photon_shot_noise_list,
        "sky_noise": sky_noise_list,
        "read_noise": read_noise_list,
        "dc_noise": dc_noise_list,
        "N": N_list,
        "RNS": RNS_list
    }
    file_name = f"rms_mags_{filename.replace('.fits', '')}_{bin_size}.json"
    output_path = os.path.join(os.getcwd(), file_name)
    with open(output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)


if __name__ == "__main__":
    main()


