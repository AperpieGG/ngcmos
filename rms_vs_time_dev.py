#!/usr/bin/env python
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from astropy.io import fits
from utils import plot_images, get_rel_phot_files, read_phot_file, bin_time_flux_error


def search_and_extract_info(filename, tic_id):
    print(f"Searching for TIC ID {tic_id} in file: {filename}")
    data_table = fits.getdata(filename)

    index = None
    for i, id in enumerate(data_table['TIC_ID']):
        if id == tic_id:
            index = i
            break

    if index is None:
        print(f"TIC ID {tic_id} not found in the data of file: {filename}")
        return None, None, None, None, None

    star_time = data_table['Time_JD'][index]
    star_flux = data_table['Relative_Flux'][index]
    tmag = data_table['Tmag'][index]
    rms = data_table['RMS'][index]
    airmass = data_table['Airmass'][index]

    print(f"Found TIC ID {tic_id} with Tmag={tmag}, RMS={rms}, Airmass={airmass} in file: {filename}")

    return star_time, star_flux, tmag, rms, airmass


def plot_rms_time(table, num_stars, tic_id=None):
    print("Plotting RMS time")

    filtered_table = table[(table['Tmag'] >= 9.2) & (table['Tmag'] <= 9.6)]
    unique_tmags = np.unique(filtered_table['Tmag'])
    print('The bright stars are: ', len(unique_tmags))

    average_rms_values = []
    times_binned = []
    max_binning = 151

    num_stars_used = 0
    num_stars_excluded = 0

    for Tmag in unique_tmags:
        # Get data for the current Tmag
        Tmag_data = table[table['Tmag'] == Tmag]

        all_jd_mid = []
        all_flux = []
        all_fluxerr = []

        for row in Tmag_data:
            tic_id_row = row['TIC_ID']
            if tic_id is not None and tic_id_row != tic_id:
                continue

            jd_mid = row['Time_JD']
            flux = row['Relative_Flux']
            fluxerr = row['Relative_Flux_err']

            # Ensure jd_mid, flux, and fluxerr are handled as arrays
            jd_mid = np.asarray(jd_mid)
            flux = np.asarray(flux)
            fluxerr = np.asarray(fluxerr)

            all_jd_mid.append(jd_mid)
            all_flux.append(flux)
            all_fluxerr.append(fluxerr)

        for jd_mid, flux, fluxerr in zip(all_jd_mid, all_flux, all_fluxerr):
            RMS_values = []
            time_seconds = []
            for i in range(1, max_binning):
                time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, flux, fluxerr, i)
                exposure_time_seconds = i * 10  # 10 seconds per binning
                RMS = np.std(dt_flux_binned)
                RMS_values.append(RMS)
                time_seconds.append(exposure_time_seconds)
            else:
                print('Using star with Tmag = {:.2f} and RMS = {:.4f}'.
                      format(Tmag, RMS_values[0]))

            num_stars_used += 1
            average_rms_values.append(RMS_values)
            times_binned.append(time_seconds)

            if num_stars_used >= num_stars:
                break

    if not average_rms_values:
        print("No stars found. Skipping this photometry file.")
        return

    print('The bright stars are: {}, Stars used: {}, Stars excluded: {}'.format(
        len(unique_tmags), num_stars_used, num_stars_excluded))

    average_rms_values = np.mean(average_rms_values, axis=0) * 1000000  # Convert to ppm

    binning_times = [i for i in range(1, max_binning)]
    RMS_model = average_rms_values[0] / np.sqrt(binning_times)

    plt.figure(figsize=(6, 10))
    plt.plot(times_binned[0], average_rms_values, 'o', color='blue', label='Actual RMS')
    plt.plot(times_binned[0], RMS_model, '--', color='red', label='Model RMS')
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


def run_for_one(phot_file, tic_id=None):
    print(f"Running for TIC ID: {tic_id} on file: {phot_file}")
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    plot_rms_time(phot_table, 5, tic_id)


def run_for_more(phot_file, num_stars):
    print(f"Running for {num_stars} stars on file: {phot_file}")
    plot_images()
    current_night_directory = '.'
    phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))
    plot_rms_time(phot_table, num_stars)


if __name__ == "__main__":
    current_night_directory = '.'

    phot_files = get_rel_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    parser = argparse.ArgumentParser(description='Plot light curve for a specific TIC ID')
    parser.add_argument('--num_stars', type=int, default=0, help='Number of stars to plot')
    parser.add_argument('--tic_id', type=int, help='Plot the time vs. binned RMS for a particular star')
    args = parser.parse_args()

    if args.tic_id is not None:
        for phot_file in phot_files:
            star_time, star_flux, tmag, rms, airmass = search_and_extract_info(phot_file, args.tic_id)
            if star_time is not None:
                print(f"Found TIC ID {args.tic_id} in file {phot_file}")
                plt.figure(figsize=(8, 6))
                plt.plot(star_time, star_flux, 'o', label=f'RMS = {rms:.4f}')
                plt.xlabel('Time (JD)')
                plt.ylabel('Relative Flux (e-)')
                plt.ylim(0.95, 1.05)
                plt.title(f'Relative Photometry for TIC ID {args.tic_id} (Tmag = {tmag:.2f})')
                plt.legend()
                plt.tight_layout()
                plt.show()
                break  # Stop after finding the first matching TIC ID
    else:
        for phot_file in phot_files:
            run_for_more(phot_file, args.num_stars)