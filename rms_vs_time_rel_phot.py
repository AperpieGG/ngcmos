#! /usr/bin/env python
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, get_rel_phot_files, read_phot_file, bin_time_flux_error


def plot_rms_time(table, num_stars=None, tic_id=None, lower_limit=0, upper_limit=20, EXPOSURE=10, bin_factor=60):
    # Filter based on TIC_ID if specified
    if tic_id is not None:
        table = table[table['TIC_ID'] == tic_id]
        if len(table) == 0:
            print(f"No data found for TIC ID {tic_id}.")
            return
        print(f"Plotting RMS for single star with TIC ID {tic_id}")
    else:
        # Filter for Tmag range
        table = table[(table['Tmag'] >= lower_limit) & (table['Tmag'] <= upper_limit)]
        unique_tmags = np.unique(table['Tmag'])
        print('Total stars in brightness range:', len(unique_tmags))

        # Calculate initial RMS per star
        stars_rms_list = []
        for Tmag in unique_tmags:
            Tmag_data = table[table['Tmag'] == Tmag]
            initial_rms = np.std(Tmag_data['Relative_Flux'])
            stars_rms_list.append((Tmag_data, initial_rms))

        # Sort and select stars with lowest RMS values
        sorted_stars = sorted(stars_rms_list, key=lambda x: x[1])[:num_stars]
        if not sorted_stars:
            print("No stars found in specified range.")
            return
        print(f'Selected {len(sorted_stars)} stars with lowest RMS values.')

    # Prepare arrays
    average_rms_values, times_binned = [], []
    max_binning = bin_factor
    star_data = sorted_stars if tic_id is None else [(table, None)]

    for Tmag_data, initial_rms in star_data:
        if 'Time_BJD' in Tmag_data.dtype.names:
            jd_mid = Tmag_data['Time_BJD']
        elif 'Time_JD' in Tmag_data.dtype.names:
            jd_mid = Tmag_data['Time_JD']
        else:
            raise ValueError("Neither 'Time_BJD' nor 'Time_JD' found in Tmag_data columns.")
        rel_flux = Tmag_data['Relative_Flux']
        rel_fluxerr = Tmag_data['Relative_Flux_err']
        RMS_values, time_seconds = [], []

        for i in range(1, max_binning):
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
            RMS_values.append(np.std(dt_flux_binned))
            time_seconds.append(i * EXPOSURE)

        average_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

        print(
            f'Star TIC_ID = {Tmag_data["TIC_ID"].iloc[0]}, Tmag = {Tmag_data["Tmag"].iloc[0]}, Initial RMS = {initial_rms:.4f}')

    # Check if data is empty
    if not average_rms_values:
        print("No stars with valid data found for plotting.")
        return

    # Calculate mean RMS values across stars
    mean_rms = np.mean(average_rms_values, axis=0) * 1e6  # ppm conversion
    binning_times = np.arange(1, max_binning)
    rms_model = mean_rms[0] / np.sqrt(binning_times)

    # Plot RMS over time
    plt.figure(figsize=(6, 10))
    plt.plot(times_binned[0], mean_rms, 'o', label='Actual RMS')
    plt.plot(times_binned[0], rms_model, '--', label='Expected RMS Model')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposure time (s)')
    plt.ylabel('RMS (ppm)')

    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.gca().tick_params(axis='y', which='minor', length=4)

    plt.legend()
    plt.tight_layout()
    plt.show()


def run_for_one(phot_file, tic_id, EXPOSURE, bin_factor):
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    plot_rms_time(phot_table, tic_id=tic_id, lower_limit=0, upper_limit=20, EXPOSURE=EXPOSURE, bin_factor=bin_factor)


def run_for_more(phot_file, num_stars, lower_limit, upper_limit, EXPOSURE, bin_factor):
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    plot_rms_time(phot_table, num_stars=num_stars, lower_limit=lower_limit, upper_limit=upper_limit,
                  EXPOSURE=EXPOSURE, bin_factor=bin_factor)


if __name__ == "__main__":
    current_night_directory = '.'
    phot_files = get_rel_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID or multiple stars')
    parser.add_argument('--num_stars', type=int, help='Number of stars to plot (only if tic_id is not provided)')
    parser.add_argument('--tic_id', type=int, help='Plot the time vs. binned RMS for a particular star')
    parser.add_argument('--bl', type=float, default=10.5, help='Lower limit for Tmag')
    parser.add_argument('--fl', type=float, default=11.5, help='Upper limit for Tmag')
    parser.add_argument('--exp', type=float, default=10.0, help='Exposure time in seconds')
    parser.add_argument('--bin', type=float, default=60, help='Maximum binning time in seconds')
    args = parser.parse_args()

    if args.tic_id is not None:
        for phot_file in phot_files:
            run_for_one(phot_file, args.tic_id, args.exp, args.bin)
    elif args.num_stars is not None:
        for phot_file in phot_files:
            run_for_more(phot_file, args.num_stars, args.bl, args.fl, args.exp, args.bin)
    else:
        print("Please specify either --tic_id for a single star or --num_stars for multiple stars.")