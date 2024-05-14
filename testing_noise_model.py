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
from utils import (read_phot_file, noise_sources, extract_airmass_zp)

APERTURE = 6
READ_NOISE = 1.56
DARK_CURRENT = 1.6


def rms_vs_mags(table, num_stars):
    """
    Calculate the mean flux and RMS for a given number of stars

    Parameters
    ----------
    table : astropy.table.Table
        Table containing the photometry data
    num_stars : int
        Number of stars to plot

    Returns
    -------
    mean_flux_list : list
        values of mean fluxes
    RMS_list : list
        values of RMS values
    sky_list : list
        values of sky values
    Tmags_list : list
        values of Tmag
    """
    mean_fluxes_list = []
    RMS_list = []
    sky_list = []
    Tmags_list = []

    unique_tic_ids = np.unique(table['TIC_ID'])
    for tic_id in unique_tic_ids[:num_stars]:  # Selecting the first num_stars unique TIC IDs
        tic_id_data = table[table['TIC_ID'] == tic_id]
        Tmag = tic_id_data['Tmag']
        relative_flux = tic_id_data['Relative_Flux']
        sky = tic_id_data['Sky']
        rms = tic_id_data['RMS']
        zero_point = tic_id_data['Zero_Point']
        airmass = tic_id_data['Airmass']

        # Calculate mean flux and RMS
        mean_fluxes_list.append(np.mean(relative_flux))
        RMS_list.append(np.nanstd(rms) * 1000000)  # Convert to ppm
        sky_list.append(np.median(sky))
        Tmags_list.append(Tmag)

    return mean_fluxes_list, RMS_list, sky_list, Tmags_list, zero_point, airmass


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC_ID')
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

    # Get the maximum number of stars based on unique TIC IDs
    max_num_stars = len(np.unique(phot_table['TIC_ID']))

    # Calculate mean and RMS for the noise model
    mean_flux_list, RMS_list, sky_list, Tmags_list, zero_point, airmass = rms_vs_mags(
        phot_table, num_stars=max_num_stars)

    # Get noise sources
    synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS = (
        noise_sources(sky_list, bin_size, airmass, zero_point, aper=APERTURE, read_noise=READ_NOISE,
                      dark_current=DARK_CURRENT))

    synthetic_mag_list = synthetic_mag.tolist()
    photon_shot_noise_list = photon_shot_noise.tolist()
    sky_noise_list = sky_noise.tolist()
    read_noise_list = read_noise.tolist()
    dc_noise_list = dc_noise.tolist()
    N_list = N.tolist()
    RNS_list = RNS.tolist()
    Tmags_list = [float(x) for x in Tmags_list]

    # Save RMS_list, and other lists to a JSON file
    output_data = {
        "TIC_IDs": np.unique(phot_table['TIC_ID']).tolist(),  # [int(x) for x in TIC_IDs
        "RMS_list": RMS_list,
        "Tmag_list": Tmags_list,
        "synthetic_mag": synthetic_mag_list,
        "photon_shot_noise": photon_shot_noise_list,
        "sky_noise": sky_noise_list,
        "read_noise": read_noise_list,
        "dc_noise": dc_noise_list,
        "N": N_list,
        "RNS": RNS_list
    }
    cwd_last_four = os.getcwd()[-4:]
    file_name = f"rms_mags_{filename.replace('.fits', '')}_{bin_size}_{cwd_last_four}.json"
    output_path = os.path.join(os.getcwd(), file_name)
    with open(output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)


if __name__ == "__main__":
    main()


