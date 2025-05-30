#!/usr/bin/env python
import itertools
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from itertools import product
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from utils import plot_images, read_phot_file, bin_time_flux_error, \
    remove_outliers, bin_by_time_interval, calc_noise, get_phot_files, target_info


plot_images()


def find_comp_star_rms(comp_fluxes, airmass):
    comp_star_rms = []
    for i, flux in enumerate(comp_fluxes):
        airmass_cs = np.polyfit(airmass, flux, 1)
        airmass_mod = np.polyval(airmass_cs, airmass)
        flux_corrected = flux / airmass_mod
        flux_norm = flux_corrected / np.median(flux_corrected)
        rms_val = np.std(flux_norm)
        comp_star_rms.append(rms_val)
    return np.array(comp_star_rms)


def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0, sig_level=1.5, dmag=0.5):
    # Calculate initial RMS of comparison stars
    comp_star_rms = find_comp_star_rms(comp_fluxes, airmass)
    print(f"Initial number of comparison stars: {len(comp_star_rms)}")

    comp_star_mask = np.ones(len(comp_star_rms), dtype=bool)
    i = 0

    while True:
        i += 1
        comp_mags = comp_mags0[comp_star_mask]
        comp_rms = comp_star_rms[comp_star_mask]
        N1 = len(comp_mags)

        if N1 == 0:
            print("No valid comparison stars left. Exiting.")
            break

        edges = np.arange(comp_mags.min(), comp_mags.max() + dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1] + edges[1:]) / 2.

        # Calculate median RMS per bin
        std_medians = np.array([np.median(comp_rms[dig == j]) if len(comp_rms[dig == j]) > 0 else np.nan
                                for j in range(1, len(edges))])

        # Remove NaNs from std_medians and mag_nodes
        valid_mask = ~np.isnan(std_medians)
        mag_nodes = mag_nodes[valid_mask]
        std_medians = std_medians[valid_mask]

        # Handle too few points for fitting
        if len(mag_nodes) < 4:
            if len(mag_nodes) > 1:
                mod = np.interp(comp_mags, mag_nodes, std_medians)
                mod0 = np.interp(comp_mags0, mag_nodes, std_medians)
            else:
                print("Not enough points for fitting. Exiting.")
                break
        else:
            spl = Spline(mag_nodes, std_medians, k=3)
            mod = spl(comp_mags)
            mod0 = spl(comp_mags0)

        std = np.std(comp_rms - mod)
        comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        N2 = np.sum(comp_star_mask)

        print(f"Iteration {i}: Stars included: {N2}, Stars excluded: {N1 - N2}")

        # Exit if the number of stars doesn't change or too many iterations
        if N1 == N2 or i > 11:
            break

    print(f'RMS of comparison stars after filtering: {len(comp_star_rms[comp_star_mask])}')
    # print(f'RMS values after filtering: {comp_star_rms[comp_star_mask]}')

    return comp_star_mask, comp_star_rms, i


def limits_for_comps(table, tic_id_to_plot, APERTURE, dmb, dmf, crop_size, color_lim):
    # Get target star info including the mean flux
    # Get target star info including the mean flux
    target_tmag, target_color, airmass_list, target_flux_mean, _, _, _, _, _ = (
        target_info(table, tic_id_to_plot, APERTURE))

    # Filter based on color index within the tolerance
    color_index = table['gaiabp'] - table['gaiarp']
    color_mask = np.abs(color_index - target_color) <= color_lim
    color_data = table[color_mask]

    # Filter stars brighter than the target within dmb and fainter than the target within dmf
    mag_mask = (color_data['Tmag'] >= target_tmag - dmb) & (color_data['Tmag'] <= target_tmag + dmf)
    valid_color_mag_table = color_data[mag_mask]

    # Exclude stars with Tmag less than 9.4 and remove the target star from the table
    valid_color_mag_table = valid_color_mag_table[valid_color_mag_table['Tmag'] > 9.4]
    filtered_table = valid_color_mag_table[valid_color_mag_table['tic_id'] != tic_id_to_plot]

    if crop_size:
        # Get target star coordinates
        x_target = table[table['tic_id'] == tic_id_to_plot]['x'][0]
        y_target = table[table['tic_id'] == tic_id_to_plot]['y'][0]

        # Apply crop filter based on coordinates
        x_min, x_max = x_target - crop_size // 2, x_target + crop_size // 2
        y_min, y_max = y_target - crop_size // 2, y_target + crop_size // 2

        # Further filter the comparison stars based on the crop region
        filtered_table = filtered_table[
            (filtered_table['x'] >= x_min) & (filtered_table['x'] <= x_max) &
            (filtered_table['y'] >= y_min) & (filtered_table['y'] <= y_max)
            ]

    return filtered_table, airmass_list


def find_best_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size, color_lim):
    # Filter the table based on color/magnitude tolerance
    filtered_table, airmass = limits_for_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size, color_lim)
    # Remove bad comparison stars

    tic_ids = np.unique(filtered_table['tic_id'])
    print(f'Number of comparison stars after filtering: {len(tic_ids)}')

    comp_fluxes = []
    comp_mags = []
    valid_tic_ids = []

    for tic_id in tic_ids:
        flux = filtered_table[filtered_table['tic_id'] == tic_id][f'flux_{APERTURE}']
        tmag = filtered_table[filtered_table['tic_id'] == tic_id]['Tmag'][0]

        comp_fluxes.append(flux)
        comp_mags.append(tmag)
        valid_tic_ids.append(tic_id)

    # Find the maximum length of flux arrays
    max_length = max(len(f) for f in comp_fluxes)

    # Filter out entries where flux arrays are shorter than max_length
    filtered_data = [(f, m, t) for f, m, t in zip(comp_fluxes, comp_mags, valid_tic_ids) if len(f) == max_length]

    # Unzip filtered data back into separate lists
    comp_fluxes, comp_mags, tic_ids = zip(*filtered_data)

    # Convert lists to arrays for further processing
    comp_fluxes = np.array(comp_fluxes)
    comp_mags = np.array(comp_mags)
    tic_ids = np.array(tic_ids)

    # Check if comp_mags is non-empty before proceeding
    if len(comp_mags) == 0:
        raise ValueError("No valid comparison stars found after filtering for flux and magnitude.")

    # Call the function to find bad comparison stars
    print(f'The dimensions of these two are: {comp_mags.shape}, {comp_fluxes.shape}')
    comp_star_mask, comp_star_rms, iterations = find_bad_comp_stars(comp_fluxes, airmass, comp_mags)

    # Filter the table based on the mask
    # print(f'Star with the min rms: {np.min(comp_star_rms)} and tic_id: {tic_ids[np.argmin(comp_star_rms)]}')

    # Filter tic_ids based on the mask
    good_tic_ids = tic_ids[comp_star_mask]

    # Now filter the table based on these tic_ids
    good_comp_star_table = filtered_table[np.isin(filtered_table['tic_id'], good_tic_ids)]

    return good_comp_star_table, airmass  # Return the filtered table including only good comp stars


def run_photometry(tic_id, dmb, dmf, crop, color_lim):
    try:
        # (All your main() logic moved here with 'args.dmb' -> dmb etc.)
        # Skip CLI parser and set values directly
        cam = 'CMOS'
        if cam == 'CMOS':
            APERTURE = 5
            DC = 1.6
            GAIN = 1.13
            EXPOSURE = 10.0
            RN = 1.56
        else:
            APERTURE = 4
            GAIN = 2
            DC = 0.00515
            EXPOSURE = 10.0
            RN = 12.9

        directory = '.'
        phot_file = get_phot_files(directory)[0]
        phot_table = read_phot_file(os.path.join(directory, phot_file))

        if tic_id not in np.unique(phot_table['tic_id']):
            return np.inf  # or None

        best_comps_table, AIRMASS = find_best_comps(phot_table, tic_id, APERTURE, dmb, dmf, crop, color_lim)
        tic_ids = np.unique(best_comps_table['tic_id'])

        time_list = []
        flux_list = []
        fluxerr_list = []
        sky_list = []

        for tic_id_ in tic_ids:
            comp_time = best_comps_table[best_comps_table['tic_id'] == tic_id_]['jd_bary']
            comp_fluxes = best_comps_table[best_comps_table['tic_id'] == tic_id_][f'flux_{APERTURE}']
            comp_fluxerrs = best_comps_table[best_comps_table['tic_id'] == tic_id_][f'fluxerr_{APERTURE}']
            comp_sky = (phot_table[phot_table['tic_id'] == tic_id_][f'flux_w_sky_{APERTURE}'] -
                        phot_table[phot_table['tic_id'] == tic_id_][f'flux_{APERTURE}'])

            time_list.append(np.array(comp_time))
            flux_list.append(np.array(comp_fluxes))
            fluxerr_list.append(np.array(comp_fluxerrs))
            sky_list.append(np.array(comp_sky))

        reference_fluxes = np.sum(flux_list, axis=0)
        comp_errs = np.vstack(([calc_noise(APERTURE, EXPOSURE, DC, GAIN, RN, AIRMASS, cfi + csi)
                                for cfi, csi in zip(flux_list, sky_list)]))
        reference_fluxerrs = np.sqrt(np.sum(comp_errs ** 2, axis=0))

        target_star = phot_table[phot_table['tic_id'] == tic_id]
        target_flux = target_star[f'flux_{APERTURE}']
        target_fluxerr = target_star[f'fluxerr_{APERTURE}']
        target_sky = target_star[f'flux_w_sky_{APERTURE}'] - target_star[f'flux_{APERTURE}']
        target_time = target_star['jd_bary']
        target_err = calc_noise(APERTURE, EXPOSURE, DC, GAIN, RN, AIRMASS, target_flux + target_sky)

        flux_ratio = target_flux / reference_fluxes
        flux_ratio_mean = np.median(flux_ratio)
        target_fluxes_dt = flux_ratio / flux_ratio_mean

        err_factor = np.sqrt((target_err / target_flux) ** 2 + (reference_fluxerrs / reference_fluxes) ** 2)
        flux_err = flux_ratio * err_factor
        target_flux_err_dt = flux_err / flux_ratio_mean

        target_time, target_fluxes_dt, target_flux_err_dt, _, _ = (
            remove_outliers(target_time, target_fluxes_dt, target_flux_err_dt))

        target_time_binned, target_fluxes_binned, target_fluxerrs_binned = (
            bin_by_time_interval(target_time, target_fluxes_dt, target_flux_err_dt, 30))

        RMS_binned = np.std(target_fluxes_binned)*1e6
        return RMS_binned
    except Exception as e:
        print(f"Error for dmb={dmb}, dmf={dmf}, crop={crop}, color_lim={color_lim}: {e}")
        return np.inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize photometric parameters.")
    parser.add_argument('--tic_id', type=int, required=True, help="Target TIC ID")
    args = parser.parse_args()

    tic_id = args.tic_id

    dmb_range = [0.1, 0.5]
    dmf_range = np.arange(0.5, 4, 0.5).round(2).tolist()
    crop_range = [None, 400, 600, 800, 1000, 1200, 1400, 2000]
    color_lim_range = np.arange(0.1, 0.8, 0.1).round(2).tolist()

    target_rms = 600
    tolerance = 200
    best_rms = np.inf
    best_params = None
    found_optimal = False

    for dmb, dmf, crop, color_lim in itertools.product(dmb_range, dmf_range, crop_range, color_lim_range):
        rms = run_photometry(tic_id, dmb, dmf, crop, color_lim)
        print(f"Params: dmb={dmb}, dmf={dmf}, crop={crop}, color_lim={color_lim} => RMS: {rms}")

        if np.abs(rms - target_rms) <= tolerance:
            print(f"\n🎯 Found optimal config! RMS = {rms}")
            print(f"Params => dmb: {dmb}, dmf: {dmf}, crop: {crop}, color_lim: {color_lim}")
            found_optimal = True
            with open("best_params_log.txt", "a") as f:
                f.write(
                    f"rel_dev_dev.py {tic_id} --dmb {dmb} --dmf {dmf} --crop {crop} --color {color_lim}  # Found RMS: {rms:.2e}\n")
            break

        if rms < best_rms:
            best_rms = rms
            best_params = (dmb, dmf, crop, color_lim)

    if not found_optimal and best_params is not None:
        print(f"\n🔍 Best RMS found (not within target tolerance): {best_rms}")
        print(f"Best parameters: dmb={best_params[0]}, dmf={best_params[1]}, crop={best_params[2]}, "
              f"color_lim={best_params[3]}")

        with open("best_params_log.txt", "a") as f:
            f.write(f"rel_dev_dev.py {tic_id} --dmb {best_params[0]} --dmf {best_params[1]} "
                    f"--crop {best_params[2]} --color {best_params[3]}  # Best RMS: {best_rms:.2e}\n")
    elif not found_optimal:
        print("⚠️ No valid RMS found.")