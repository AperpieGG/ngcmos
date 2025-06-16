#!/usr/bin/env python

import argparse
import os
import json
import numpy as np
from utils import noise_sources, bin_time_flux_error


class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types """

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
    cam = args.cam

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

    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'targets')
    json_files = [f for f in os.listdir(target_dir) if f.startswith('target_light_curve_') and f.endswith('.json')]
    print(f"Found {len(json_files)} target JSON files in {target_dir}")

    zp = extract_zero_point(f'zp{APERTURE}.json')

    RMS_list, Tmags_list, color_list = [], [], []
    photon_shot_noise_list = []
    sky_noise_list = []
    read_noise_list = []
    dc_noise_list = []
    N_list = []
    RNS_list = []
    tic_ids = []

    sky_values = []
    airmass_values = []

    for file in json_files:
        file_path = os.path.join(target_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        tic_id = data['TIC_ID']
        tic_ids.append(tic_id)

        Tmag = float(data['Tmag'])
        color = float(data['COLOR'])
        sky = np.mean(np.array(data['Sky']))
        airmass = np.mean(np.array(data['Airmass']))
        sky_values.append(sky)
        airmass_values.append(airmass)

        flux = np.array(data['Relative_Flux'])
        flux_err = np.array(data['Relative_Flux_err'])
        time = np.array(data['Time_BJD'])

        if bin_size > 1:
            time, flux, flux_err = bin_time_flux_error(time, flux, flux_err, bin_fact=bin_size)
            RMS = np.std(flux) * 1e6
        else:
            RMS = float(data['RMS']) * 1e6

        RMS_list.append(RMS)
        Tmags_list.append(Tmag)
        color_list.append(color)

        # Per-star noise values still calculated for diagnostics
        _, psn, sn, rn, dcn, N, RNS = noise_sources(
            [sky], bin_size, [airmass], zp, APERTURE,
            READ_NOISE, DARK_CURRENT, EXPOSURE, GAIN
        )

        photon_shot_noise_list.append(psn[0])
        sky_noise_list.append(sn[0])
        read_noise_list.append(rn[0])
        dc_noise_list.append(dcn[0])
        N_list.append(N[0])
        RNS_list.append(RNS[0])

    # Compute global synthetic magnitude once
    avg_sky = np.mean(sky_values)
    avg_airmass = np.mean(airmass_values)
    print("Average sky flux:", avg_sky)

    synthetic_mag, _, _, _, _, _, _ = noise_sources(
        [avg_sky], bin_size, [avg_airmass], zp, APERTURE,
        READ_NOISE, DARK_CURRENT, EXPOSURE, GAIN
    )

    output_data = {
        "TIC_IDs": tic_ids,
        "RMS_list": RMS_list,
        "Tmag_list": Tmags_list,
        "synthetic_mag": synthetic_mag[0],
        "photon_shot_noise": photon_shot_noise_list,
        "sky_noise": sky_noise_list,
        "read_noise": read_noise_list,
        "dc_noise": dc_noise_list,
        "N": N_list,
        "RNS": RNS_list,
        "COLOR": color_list
    }

    cwd_last_four = current_dir[-4:]
    output_filename = f"rms_mags_from_json_{cam}_{cwd_last_four}.json"

    with open(output_filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=4, cls=NumpyEncoder)

    print(f"✅ Results saved to {output_filename}")


if __name__ == '__main__':
    main()