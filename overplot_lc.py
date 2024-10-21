#!/usr/bin/env python
import json
import matplotlib.pyplot as plt
import glob
from utils import plot_images

plot_images()


def plot_light_curves():
    # Search for all JSON files that start with 'target_light_curve'
    json_files = glob.glob('target_light_curve*.json')

    if not json_files:
        print('No JSON files found that start with "target_light_curve".')
        return

    plt.figure(figsize=(10, 6))

    for json_filename in json_files:
        # Load the JSON data from each file
        with open(json_filename, 'r') as json_file:
            data = json.load(json_file)

        # Extract time and flux data
        target_time_binned = data['time']
        target_fluxes_dt = data['flux']

        # Determine if the file corresponds to 'CMOS' or 'CCD' based on the filename
        if json_filename.endswith('CMOS.json'):
            label = 'CMOS'
        elif json_filename.endswith('CCD.json'):
            label = 'CCD'
        else:
            label = 'Unknown'  # Fallback if neither matches

        # Plot each light curve
        plt.plot(target_time_binned, target_fluxes_dt, 'o-', label=f'{label}')

    # Customize the plot
    plt.title('Target Light Curves')
    plt.xlabel('Binned Time (JD)')
    plt.ylabel('Normalized Flux')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    plot_light_curves()