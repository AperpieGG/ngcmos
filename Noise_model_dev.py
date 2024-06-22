import argparse
import os
import json
from collections import defaultdict
import numpy as np
from astropy.io import fits
from utils import noise_sources  # Assuming you have a noise_sources function in utils

# Constants for noise calculations
APERTURE = 6
READ_NOISE = 1.56
DARK_CURRENT = 1.6

class NumpyEncoder(json.JSONEncoder):
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
    data_dict = defaultdict(list)

    try:
        with fits.open(filename) as hdul:
            data = hdul[1].data  # Assuming data is in the first extension
            for row in data:
                tic_id = row['TIC_ID']
                rms = row['RMS'] * 1000000  # Multiply RMS by 1000000 to convert to ppm
                sky = row['Sky']
                Tmag = row['Tmag']
                airmass = row['Airmass']
                zp = row['ZP']
                mag = row['Magnitude']

                data_dict[tic_id].append({
                    'RMS': rms,
                    'Sky': sky,
                    'Tmag': Tmag,
                    'Airmass': airmass,
                    'ZP': zp,
                    'Magnitude': mag
                })
    except Exception as e:
        print(f"Error reading FITS file: {e}")

    return data_dict


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Read and organize TIC IDs with associated RMS, Sky, Airmass, ZP, and Magnitude from FITS table')
    parser.add_argument('filename', type=str, help='Name of the FITS file containing photometry data')
    args = parser.parse_args()
    filename = args.filename

    # Get the current working directory
    current_dir = os.getcwd()

    # Construct full path to the FITS file
    file_path = os.path.join(current_dir, filename)

    # Read and organize data by TIC ID
    data_dict = read_data(file_path)

    # Prepare lists for noise_sources function
    sky_list = []
    airmass_list = []
    zp = []

    # Calculate RMS_list, sky_list, Tmags_list, and other lists
    RMS_list = []
    mags_list = []
    Tmags_list = []
    for tic_id, data_list in data_dict.items():
        for entry in data_list:
            RMS_list.append(entry['RMS'])
            sky_list.append(entry['Sky'])
            Tmags_list.append(entry['Tmag'])
            airmass_list.append(entry['Airmass'])
            zp.append(entry['ZP'])
            mags_list.append(entry['Magnitude'])

    # Get noise sources
    bin_size = 1  # Example bin size, adjust as needed
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
    Tmags_list = [float(x) for x in Tmags_list]
    mags_list = [float(x) for x in mags_list]

    # Save results to JSON file
    output_data = {
        "TIC_IDs": list(data_dict.keys()),  # TIC IDs
        "RMS_list": RMS_list,
        "Tmag_list": Tmags_list,
        "mags_list": mags_list,
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
        json.dump(output_data, json_file, indent=4, cls=NumpyEncoder)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()