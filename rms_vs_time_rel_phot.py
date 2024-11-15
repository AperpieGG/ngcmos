#! /usr/bin/env python
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from utils import plot_images, get_rel_phot_files, read_phot_file, bin_time_flux_error


def plot_rms_time(table, num_stars=None, tic_id=None, lower_limit=0, upper_limit=20, EXPOSURE=10, bin_factor=60):
    # If a specific TIC ID is provided, filter for that star
    if tic_id is not None:
        table = table[table['TIC_ID'] == tic_id]
        if len(table) == 0:
            print(f"No data found for TIC ID {tic_id}.")
            return
        print(f"Plotting RMS for single star with TIC ID {tic_id}")
    else:
        # Filter for the specified Tmag range for multiple stars
        table = table[(table['Tmag'] >= lower_limit) & (table['Tmag'] <= upper_limit)]
        unique_tmags = np.unique(table['Tmag'])
        print('Total stars in brightness range:', len(unique_tmags))

        # Initialize arrays to hold star data
        stars_rms_list = []

        # Calculate the initial RMS for each star in the filtered range
        for Tmag in unique_tmags:
            Tmag_data = table[table['Tmag'] == Tmag]
            rel_flux = Tmag_data['Relative_Flux']
            initial_rms = np.std(rel_flux)
            stars_rms_list.append((Tmag_data, initial_rms))

        # Sort stars by initial RMS and select top `num_stars`
        sorted_stars = sorted(stars_rms_list, key=lambda x: x[1])[:num_stars]
        print(f'Selected {len(sorted_stars)} stars with lowest RMS values.')

    # Prepare data for plotting
    average_rms_values = []
    times_binned = []
    max_binning = int(bin_factor)

    # Select stars for plotting based on `num_stars` or single star with `tic_id`
    star_data = sorted_stars if tic_id is None else [(table, None)]

    for Tmag_data, initial_rms in star_data:
        if 'Time_BJD' in Tmag_data.dtype.names:
            jd_mid = Tmag_data['Time_BJD']
        elif 'Time_JD' in Tmag_data.dtype.names:
            jd_mid = Tmag_data['Time_JD']
        else:
            raise ValueError("Neither 'Time_BJD' nor 'BJD' found in Tmag_data columns.")

        rel_flux = Tmag_data['Relative_Flux']
        rel_fluxerr = Tmag_data['Relative_Flux_err']
        current_tic_id = Tmag_data['TIC_ID'][0]

        RMS_values = []
        time_seconds = []
        for i in range(1, max_binning):
            time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
            exposure_time_seconds = i * EXPOSURE  # 10 seconds per bin
            RMS = np.std(dt_flux_binned)
            RMS_values.append(RMS)
            time_seconds.append(exposure_time_seconds)

        average_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

        print(f'Star TIC_ID = {current_tic_id}, Tmag = {Tmag_data["Tmag"][0]}, RMS = {RMS_values[0]:.4f}')

    if not average_rms_values:
        print("No stars found. Skipping this photometry file.")
        return

    # Calculate average RMS across selected stars
    average_rms_values = np.mean(average_rms_values, axis=0) * 1e6  # Convert to ppm

    # Generate binning times
    binning_times = [i for i in range(1, max_binning)]

    # Expected RMS decrease model
    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    # Plot RMS over time
    plt.figure(figsize=(6, 10))
    plt.plot(times_binned[0], average_rms_values, 'bo')
    plt.plot(times_binned[0], RMS_model, '--', color='black')
    plt.axvline(x=900, color='red', linestyle='-')
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