#!/usr/bin/env python

import argparse
import os
import json
import numpy as np
from utils import noise_sources, bin_time_flux_error


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def extract_zero_point(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            zero_point = json.load(json_file)
        print(f"Extracted zero point: {zero_point}")
        return zero_point
    except FileNotFoundError:
        print(f"File {json_file_path} not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON from file.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_size', type=int, default=1, help='Bin size for noise calculations')
    parser.add_argument('--cam', type=str, default='CMOS', help='Camera type (CMOS or CCD)')
    args = parser.parse_args()

    bin_size = args.bin_size
    cam = args.cam.upper()

    # Camera configuration
    if cam == 'CMOS':
        READ_NOISE = 1.56
        DARK_CURRENT = 1.6
        GAIN = 1.13
        APERTURE = 5
        EXPOSURE = 10.0
    else:
        READ_NOISE = 12.9
        DARK_CURRENT = 0.00515
        GAIN = 2
        APERTURE = 4
        EXPOSURE = 10.0

    # Locate JSON target files
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'targets')
    json_files = [f for f in os.listdir(target_dir) if f.startswith('target_light_curve_') and f.endswith('.json')]
    print(f"📁 Found {len(json_files)} JSON files in {target_dir}")

    # Load zero point
    zp = extract_zero_point(f'zp{APERTURE}.json')

    # Prepare output containers
    RMS_list, Tmags_list, color_list = [], [], []
    synthetic_mag_list = []
    photon_shot_noise_list = []
    sky_noise_list = []
    read_noise_list = []
    dc_noise_list = []
    N_list = []
    RNS_list = []
    tic_ids = []
    all_sky = []
    all_airmass = []

    for file in json_files:
        file_path = os.path.join(target_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        tic_id = data['TIC_ID']
        tic_ids.append(tic_id)

        Tmag = float(data['Tmag'])
        color = float(data['COLOR'])
        sky = np.array(data['Sky'])
        sky = np.mean(sky, axis=0)  # collapse to 1D average
        airmass = np.array(data['Airmass'])
        flux = np.array(data['Relative_Flux'])
        flux_err = np.array(data['Relative_Flux_err'])
        time = np.array(data['Time_BJD'])

        all_sky.append(sky)
        all_airmass.extend(airmass)

        # Bin and calculate RMS
        if bin_size > 1:
            time, flux, flux_err = bin_time_flux_error(time, flux, flux_err, bin_fact=bin_size)
            RMS = np.std(flux) * 1e6  # ppm
        else:
            RMS = float(data.get('RMS', np.std(flux))) * 1e6  # fallback if RMS not stored

        RMS_list.append(RMS)
        Tmags_list.append(Tmag)
        color_list.append(color)

    # Convert to arrays
    all_sky = np.array(all_sky)
    airmass_array = np.array(all_airmass)

    print("🌌 Average sky flux:", np.mean(all_sky))
    print("🧭 Calculated ZP vs Header ZP mean:", np.mean(zp))

    # Call noise_sources
    synthetic_mag, psn, sn, rn, dcn, N, RNS = noise_sources(
        all_sky, bin_size, airmass_array, zp, APERTURE, READ_NOISE, DARK_CURRENT, EXPOSURE, GAIN
    )

    # Convert all outputs to serializable lists
    output_data = {
        "TIC_IDs": tic_ids,
        "RMS_list": RMS_list,
        "Tmag_list": Tmags_list,
        "synthetic_mag": synthetic_mag.tolist(),
        "photon_shot_noise": psn.tolist(),
        "sky_noise": sn.tolist(),
        "read_noise": rn.tolist(),
        "dc_noise": dcn.tolist(),
        "N": N.tolist(),
        "RNS": RNS.tolist(),
        "COLOR": color_list
    }

    # Create output filename
    output_name = f"rms_mags_from_json_{cam}_{current_dir[-4:]}.json"
    with open(output_name, 'w') as json_file:
        json.dump(output_data, json_file, indent=4, cls=NumpyEncoder)

    print(f"✅ Noise model results saved to {output_name}")


if __name__ == "__main__":
    main()