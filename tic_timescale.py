#! /usr/bin/env python
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import bin_time_flux_error, plot_images

plot_images()


def load_json_data(json_file):
    """Load and parse JSON file containing photometry data for a single TIC_ID."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Assume JSON is a list of dicts or a single dict
    if isinstance(data, list):
        data = data[0]  # Use the first entry

    return {
        'Time_BJD': np.array(data['Time_BJD']),
        'Relative_Flux': np.array(data['Relative_Flux']),
        'RMS': np.array(data['RMS']),
        'TIC_ID': data['TIC_ID']
    }


def trim_target_data_by_time(data):
    """
    Trim the data points by removing the first 30 minutes and the last 15 minutes of data based on `Time_BJD`.
    """
    start_threshold = 57 / (24 * 60)  # 30 minutes in days
    end_threshold = 5 / (24 * 60)  # 15 minutes in days

    start_time = data['Time_BJD'][0]
    end_time = data['Time_BJD'][-1]

    mask = (data['Time_BJD'] >= (start_time + start_threshold)) & (data['Time_BJD'] <= (end_time - end_threshold))

    return {
        'Time_BJD': data['Time_BJD'][mask],
        'Relative_Flux': data['Relative_Flux'][mask],
        'RMS': np.std(data['Relative_Flux'][mask]),
        'TIC_ID': data['TIC_ID']
    }


def compute_rms_values(data, exp, max_binning):
    jd_mid = data['Time_BJD']
    rel_flux = data['Relative_Flux']
    rel_fluxerr = np.std(rel_flux) * np.ones_like(rel_flux)  # Dummy error array
    print(f'Using exposure time: {exp}')

    RMS_values = []
    time_seconds = []

    for i in range(1, max_binning):
        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
        exposure_time_seconds = i * exp
        RMS = np.std(dt_flux_binned)
        RMS_values.append(RMS)
        time_seconds.append(exposure_time_seconds)

    average_rms_values = np.array(RMS_values) * 1e6  # Convert to ppm
    RMS_model = average_rms_values[0] / np.sqrt(np.arange(1, max_binning))

    return time_seconds, average_rms_values, RMS_model


def plot_timescale(times, avg_rms, RMS_model, label, label_color):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(times, avg_rms, 'o', label=f"{label} Data", color=label_color)
    ax.plot(times, RMS_model, '--', label=f"{label} Model", color=label_color)
    ax.axvline(x=900, color='black', linestyle='-', label='Reference Line (x=900)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Exposure Time (s)')
    ax.set_ylabel('RMS (ppm)')
    ax.set_title('RMS vs Exposure Time')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.tick_params(axis='y', which='minor', length=4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and plot RMS from JSON photometry file.')
    parser.add_argument('json_file', type=str, help='Path to JSON photometry file')
    parser.add_argument('--bin', type=int, default=180, help='Maximum binning steps')
    parser.add_argument('--cam', type=str, default='CMOS', help='Maximum binning steps')

    args = parser.parse_args()

    phot_data = load_json_data(args.json_file)
    print("Trimming data by time...")
    phot_data = trim_target_data_by_time(phot_data)
    times, avg_rms, RMS_model = compute_rms_values(phot_data, exp=10, max_binning=args.bin)
    if args.cam == 'CMOS':
        label_color = 'blue'
    else:
        label_color = 'red'

    plot_timescale(times, avg_rms, RMS_model, label=str(phot_data['TIC_ID']), label_color=label_color)
