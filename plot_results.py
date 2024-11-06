#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt, ticker
import json
import numpy as np
from utils import plot_images


def load_rms_mags_data(filename):
    """
    Load RMS and magnitude data from JSON file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def mask_outliers_by_model(Tmag_list, RMS_list, color_list, synthetic_mag, RNS, deviation_factor=2):
    """
    Mask stars that have an RMS significantly higher than the model.
    Args:
        Tmag_list (list): List of magnitudes (Tmag).
        RMS_list (list): List of RMS values.
        color_list (list): List of color values.
        synthetic_mag (list): Synthetic magnitude values for the model.
        RNS (list): Model noise values (RMS as a function of magnitude).
        deviation_factor (float): Factor to define significant deviation from model.

    Returns:
        masked_indices (list): Indices of stars that deviate from the model.
    """
    # Interpolate model RMS values to match Tmag_list
    model_rms_interp = np.interp(Tmag_list, synthetic_mag, RNS)
    masked_indices = [i for i, (rms, model_rms) in enumerate(zip(RMS_list, model_rms_interp))
                      if rms > model_rms * deviation_factor]

    print("Masked Stars:")
    for i in masked_indices:
        print(f"Index: {i}, Tmag: {Tmag_list[i]}, RMS: {RMS_list[i]}, Color: {color_list[i]}")

    return masked_indices


def plot_noise_model(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    RMS_list = data['RMS_list']
    tic_ids = data['TIC_IDs']
    color_list = data['COLOR']
    Tmag_list = data['Tmag_list']
    synthetic_mag = data['synthetic_mag']
    RNS = data['RNS']
    photon_shot_noise = data['photon_shot_noise']
    read_noise = data['read_noise']
    dc_noise = data['dc_noise']
    sky_noise = data['sky_noise']
    N = data['N']
    print(f'The average scintillation noise is: {np.mean(N)}')

    # Mask stars that deviate significantly from the model
    masked_indices = mask_outliers_by_model(Tmag_list, RMS_list, color_list, synthetic_mag, RNS)

    # Exclude masked stars from the total data
    total_RMS = [RMS_list[i] for i in range(len(RMS_list)) if i not in masked_indices]
    total_mags = [Tmag_list[i] for i in range(len(Tmag_list)) if i not in masked_indices]
    total_colors = [color_list[i] for i in range(len(color_list)) if i not in masked_indices]

    # Verify sizes match
    if len(total_mags) != len(total_RMS) or len(total_mags) != len(total_colors):
        raise ValueError("Mismatch in sizes: total_mags, total_RMS, and total_colors should be the same length.")


    # Scatter plot with remaining stars
    scatter = ax.scatter(total_mags, total_RMS, c=total_colors, cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Color Scale')

    # Plot various noise sources
    ax.plot(synthetic_mag, RNS, color='black', label='total noise')
    ax.plot(synthetic_mag, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(synthetic_mag, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(synthetic_mag, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(synthetic_mag, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(synthetic_mag, np.ones(len(synthetic_mag)) * N, color='orange', label='scintillation noise',
            linestyle='--')

    # Plot formatting
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def main(json_file):
    # Set plot parameters
    plot_images()
    # Load RMS and magnitude data from JSON file
    data = load_rms_mags_data(json_file)

    # Plot RMS vs magnitudes
    plot_noise_model(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot RMS vs Magnitudes')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing RMS and magnitude data')
    args = parser.parse_args()

    # Run main function
    main(args.json_file)