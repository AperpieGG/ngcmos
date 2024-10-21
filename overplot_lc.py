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

    # Set up the figure and the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    plt.subplots_adjust(wspace=0)  # Remove space between plots

    plot_axes = [ax2, ax1]  # Swapped the axes (ax2 is now left, ax1 is right)

    for i, json_filename in enumerate(json_files[:2]):  # Limit to two plots (one per column)
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

        # Plot each light curve in black
        plot_axes[i].plot(target_time_binned, target_fluxes_dt, 'o', color='black')
        plot_axes[i].set_title(f'{label}')
        plot_axes[i].set_xlabel('Binned Time (BJD)')
        if i == 1:
            plot_axes[i].set_ylabel('Normalized Flux')
        plot_axes[i].grid(True)

    plt.show()


if __name__ == "__main__":
    plot_light_curves()