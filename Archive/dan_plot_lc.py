#!/usr/bin/env python
"""
Plot light curve for a given TIC ID from a FITS file containing photometry data
The data is taken from the rel_phot_NGFIELD.fits file that is created from relative_phot.py
This script is plotting the unbinned relative photometry lightcurve, and mask the area with airmass greater than 1.75
This script is to test the red noise from timescale of noise script.
"""
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from utils import plot_images, bin_time_flux_error, get_rel_phot_files, read_phot_file


def plot_lc(filename, tic_id_to_plot, bin_size):
    with fits.open(filename) as hdul:
        table = hdul[1].data

    tic_id_data = table[table['TIC_ID'] == tic_id_to_plot]

    if len(tic_id_data) == 0:
        print(f"TIC ID {tic_id_to_plot} not found in the data.")
        return

    tmag = tic_id_data['Tmag'][0]

    if 'Time_JD' not in tic_id_data.names:
        time = tic_id_data['Time_BJD']
    else:
        time = tic_id_data['Time_JD']

    flux = tic_id_data['Relative_Flux']
    flux_err = tic_id_data['Relative_Flux_err']

    if 'Airmass' not in tic_id_data.names:
        airmass = np.zeros_like(time)
    else:
        airmass = tic_id_data['Airmass']

    valid_mask = airmass < 1.75
    time = time[valid_mask]
    flux = flux[valid_mask]
    flux_err = flux_err[valid_mask]
    airmass = airmass[valid_mask]

    rms = np.std(flux)
    print(f'The RMS for TIC {tic_id_to_plot} is: {rms:.4f}')

    if bin_size > 1:
        time_binned, flux_binned, flux_err_binned = \
            bin_time_flux_error(time, flux, flux_err, bin_size)
        rms_binned = np.std(flux_binned)
    else:
        time_binned, flux_binned, flux_err_binned, rms_binned = time, flux, flux_err, rms

    fig, ax1 = plt.subplots(figsize=(8, 6))

    if bin_size > 1:
        ax1.plot(time, flux, 'o', color='blue', alpha=0.2)

    print(f'The data points for binned data for TIC {tic_id_to_plot}: {len(flux_binned)}')
    ax1.plot(time_binned, flux_binned, 'o', label=f'RMS = {rms_binned:.4f}', color='red')
    ax1.set_xlabel('Time (JD)')
    ax1.set_ylabel('Relative Flux')
    ax1.set_ylim(0.95, 1.05)
    ax1.set_title(f'Rel Phot for TIC ID {tic_id_to_plot} and Tmag = {tmag:.2f}')

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Airmass')

    primary_xticks = ax1.get_xticks()
    interpolated_airmass = np.interp(primary_xticks, time, airmass)
    airmass_ticks = [f'{a:.2f}' for a in interpolated_airmass]

    ax2.set_xticks(primary_xticks)
    ax2.set_xticklabels(airmass_ticks, rotation=45, ha='right')

    ax1.legend()
    plt.tight_layout()

    output_filename = f"{tic_id_to_plot}_lc.png"
    plt.savefig(output_filename, dpi=200)
    plt.close()
    print(f"Saved plot to {output_filename}")


def main():
    plot_images()
    directory = '.'

    # Define the known TIC IDs you want to plot
    tic_ids_to_plot = [4611043, 5796255, 5796320, 5796376, 169746092, 169746369, 169746459, 169763609, 169763615,
                       169763631, 169763812, 169763929, 169763985, 169764011, 169764168, 169764174, 188619865,
                       188620052, 188620343, 188620450, 188620477, 188620644, 188622237, 188622268, 188622275,
                       188622523, 188627904, 188628115, 188628237, 188628252, 188628309, 188628413, 188628448,
                       188628555, 188628748, 188628755, 214657492, 214657985, 214658021, 214661588, 214661799,
                       214661930, 214662807, 214662895, 214662905, 214664699, 214664842,
                       270185125, 270185254, 270187139, 270187208, 270187283]

    bin_size = 1  # Set the bin size here

    filenames = get_rel_phot_files(directory)

    for tic_id in tic_ids_to_plot:
        found = False
        for phot_file in filenames:
            phot_table = read_phot_file(os.path.join(directory, phot_file))

            if tic_id in phot_table['tic_id']:
                print(f'Found TIC ID {tic_id} in photometry file: {phot_file}')
                plot_lc(phot_file, tic_id, bin_size)
                found = True
                break

        if not found:
            print(f"TIC ID {tic_id} not found in any photometry file.")


if __name__ == "__main__":
    main()
