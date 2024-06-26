#!/usr/bin/env python

import argparse
import os
import json
import numpy as np
from collections import defaultdict
from astropy.io import fits
from utils import noise_sources  # Assuming you have a noise_sources function in utils

# Constants for noise calculations
APERTURE = 6  # Aperture size for the telescope
READ_NOISE = 1.56  # Read noise in electrons
DARK_CURRENT = 1.6  # Dark current in electrons per second


class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def read_data(filename):
    """
    Read data from a FITS file and return it as a structured array.

    Parameters:
    - filename: str, path to the FITS file.

    Returns:
    - data: structured array from the FITS file.
    """
    try:
        with fits.open(filename) as hdul:
            data = hdul[1].data  # Assuming data is in the first extension
    except Exception as e:
        print(f"Error reading FITS file: {e}")
        return None

    return data


def main():
    """ Main function to parse arguments, read data, calculate noise sources, and save results to a JSON file """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Read and organize TIC IDs with associated RMS, Sky, Airmass, ZP, and Magnitude from FITS table')
    parser.add_argument('filename', type=str, help='Name of the FITS file containing photometry data')
    parser.add_argument('--bin_size', type=int, default=1, help='Bin size for noise calculations')
    args = parser.parse_args()
    filename = args.filename
    bin_size = args.bin_size

    # Get the current working directory
    current_dir = os.getcwd()

    # Construct full path to the FITS file
    file_path = os.path.join(current_dir, filename)

    # Read data from the FITS file
    data = read_data(file_path)
    if data is None:
        return

    # Extract unique TIC IDs
    unique_tic_ids = np.unique(data['TIC_ID'])

    results = {}

    # Iterate over each unique TIC ID
    for tic_id in unique_tic_ids:
        tic_data = data[data['TIC_ID'] == tic_id]

        RMS_list = tic_data['RMS'] * 1000000  # Convert RMS to ppm
        sky_list = tic_data['Sky']
        Tmags_list = tic_data['Tmag']
        airmass_list = tic_data['Airmass']
        zp = tic_data['ZP']
        mags_list = tic_data['Magnitude']

        # Get noise sources
        synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS = (
            noise_sources(sky_list, bin_size, airmass_list, zp, APERTURE, READ_NOISE, DARK_CURRENT))

        # Convert lists to JSON serializable lists
        synthetic_mag_list = synthetic_mag.tolist()
        photon_shot_noise_list = photon_shot_noise.tolist()
        sky_noise_list = sky_noise.tolist()
        read_noise_list = read_noise.tolist()
        dc_noise_list = dc_noise.tolist()
        N_list = N.tolist()
        RNS_list = RNS.tolist()

        # Store results for the current TIC ID
        results[tic_id] = {
            "RMS_list": RMS_list.tolist(),
            "Tmag_list": Tmags_list.tolist(),
            "mags_list": mags_list.tolist(),
            "synthetic_mag": synthetic_mag_list,
            "photon_shot_noise": photon_shot_noise_list,
            "sky_noise": sky_noise_list,
            "read_noise": read_noise_list,
            "dc_noise": dc_noise_list,
            "N": N_list,
            "RNS": RNS_list
        }

    # Construct output file name
    cwd_last_four = os.getcwd()[-4:]
    file_name = f"rms_mags_{filename.replace('.fits', '')}_{bin_size}_{cwd_last_four}.json"
    output_path = os.path.join(current_dir, file_name)

    # Save JSON file using custom encoder
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4, cls=NumpyEncoder)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()