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

    # Limit to two JSON files (or however many are available)
    num_files = min(len(json_files), 2)
    fig, axes = plt.subplots(1, num_files, figsize=(12, 6), sharey=True)
    plt.subplots_adjust(wspace=0)  # Remove space between plots

    # Ensure `axes` is a list, even if there's only one subplot
    if num_files == 1:
        axes = [axes]

    for i, json_filename in enumerate(json_files[:num_files]):
        # Load the JSON data from each file
        with open(json_filename, 'r') as json_file:
            data = json.load(json_file)

        # Extract time and flux data
        target_time_binned = data['time']
        target_fluxes_dt = data['flux']

        # Dynamically determine label based on filename
        label = json_filename.split('_')[-1].replace('.json', '')  # Use last part before .json

        # Plot each light curve in black
        axes[i].plot(target_time_binned, target_fluxes_dt, 'o', color='black', label=f'RMS = {data["RMS"]:.4f}')
        axes[i].set_title(f'{label}')
        axes[i].set_xlabel('Binned Time (BJD)')
        if i == 0:
            axes[i].set_ylabel('Normalized Flux')
        axes[i].grid(True)
        axes[i].legend()  # Add legend to display the RMS
    plt.show()


if __name__ == "__main__":
    plot_light_curves()