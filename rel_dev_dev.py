#!/usr/bin/env python
import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import json
from astropy.visualization import ZScaleInterval
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from utils import plot_images, read_phot_file, bin_time_flux_error, \
    remove_outliers

# Constants for filtering stars
COLOR_TOLERANCE = 0.2  # Color index tolerance for comparison stars
plot_images()


def get_image_data(frame_id):
    """
    Get the image data corresponding to the given frame_id.

    Parameters:
        frame_id (str): The frame_id of the image.

    Returns:
        numpy.ndarray or None: The image data if the image exists, otherwise None.
    """
    # Define the directory where the images are stored (use cwd if not explicitly defined)
    image_directory = os.getcwd()  # You can change this to the desired image directory path
    image_path_fits = os.path.join(image_directory, frame_id)

    print(f"Looking for FITS image at: {image_path_fits}")

    # Check if the uncompressed FITS file exists
    if os.path.exists(image_path_fits):
        print("FITS file found.")
        with fits.open(image_path_fits) as hdul:
            image_data = hdul[0].data  # Assuming the image data is in the primary HDU
        return image_data


def plot_comps_position(table, tic_id_to_plot, filtered_tic_ids, camera):
    # load the fits image on this particular field.
    image_data = get_image_data(table[table['tic_id'] == tic_id_to_plot]['frame_id'][0])

    # Assuming x, y coordinates are already extracted for the target star and comparison stars
    # Example: x_target, y_target for the target star
    x_target = table[table['tic_id'] == tic_id_to_plot]['x'][0]
    y_target = table[table['tic_id'] == tic_id_to_plot]['y'][0]
    print(f'Target star coordinates: x = {x_target}, y = {y_target}')

    # Create a circle for the target star (in red)
    interval = ZScaleInterval()
    vmin, vmax = np.percentile(image_data, [5, 95])
    target_circle = plt.Circle((x_target, y_target), radius=5, color='green', fill=False, linewidth=1.5)
    plt.gca().add_patch(target_circle)

    # Do the same for comparison stars (for example, x_comp, y_comp for each comparison star)
    for tic_id in filtered_tic_ids:
        x_comp = table[table['tic_id'] == tic_id]['x'][0]
        y_comp = table[table['tic_id'] == tic_id]['y'][0]
        comp_circle = plt.Circle((x_comp, y_comp), radius=5, color='blue', fill=False, linewidth=1.5)
        plt.gca().add_patch(comp_circle)

        # Add the TIC ID label for each comparison star
        plt.text(x_comp, y_comp + 10, str(tic_id), color='blue', fontsize=10, ha='center')

    if camera == 'CMOS':
        plt.imshow(image_data, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(image_data, cmap='hot', vmin=vmin, vmax=vmax)
    plt.tight_layout()
    # Add labels and legend
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    # plt.legend([target_circle, comp_circle], ['Target', 'Comp Stars'], loc='upper right')
    plt.title(f'Target: {tic_id_to_plot}')
    plt.show()


def target_info(table, tic_id_to_plot, APERTURE):
    target_star = table[table['tic_id'] == tic_id_to_plot]  # Extract the target star data
    target_tmag = target_star['Tmag'][0]  # Extract the TESS magnitude of the target star
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]  # Extract the color index
    airmass_list = target_star['airmass']  # Extract airmass_list from target star
    # Calculate mean flux for the target star (specific to the chosen aperture)
    target_flux_mean = target_star[f'flux_{APERTURE}'].mean()

    return target_tmag, target_color_index, airmass_list, target_flux_mean


def extract_region_coordinates(region):
    # Assuming region is a dictionary that has 'x_min', 'x_max', 'y_min', 'y_max' keys
    x_min = region['x_min']
    x_max = region['x_max']
    y_min = region['y_min']
    y_max = region['y_max']

    return x_min, x_max, y_min, y_max


def limits_for_comps(table, tic_id_to_plot, APERTURE, dmb, dmf, crop_size, json_file):
    # Get target star info including the mean flux
    target_tmag, target_color, airmass_list, target_flux_mean = target_info(table, tic_id_to_plot, APERTURE)

    # Filter based on color index within the tolerance
    color_index = table['gaiabp'] - table['gaiarp']
    color_mask = np.abs(color_index - target_color) <= COLOR_TOLERANCE
    color_data = table[color_mask]

    # Filter stars brighter than the target within dmb and fainter than the target within dmf
    mag_mask = (color_data['Tmag'] >= target_tmag - dmb) & (color_data['Tmag'] <= target_tmag + dmf)
    valid_color_mag_table = color_data[mag_mask]

    # Exclude stars with Tmag less than 9.4 and remove the target star from the table
    valid_color_mag_table = valid_color_mag_table[valid_color_mag_table['Tmag'] > 9.4]
    filtered_table = valid_color_mag_table[valid_color_mag_table['tic_id'] != tic_id_to_plot]

    # Then in your main code, make these changes:
    if json_file:  # Check if json_file is set (instead of json)
        with open('fwhm_positions.json', 'r') as file:
            regions_data = json.load(file)  # Use `json.load` with the module, not the boolean

        # Extract coordinates for central and similar regions
        region_coordinates = []

        # Central region
        central = regions_data['central_region']['position']
        region_coordinates.append((central['x_start'], central['x_end'], central['y_start'], central['y_end']))

        # Similar regions
        for similar_region in regions_data['central_region']['similar_regions']:
            position = similar_region['position']
            region_coordinates.append((position['x_start'], position['x_end'], position['y_start'], position['y_end']))

        # Filter stars based on the defined regions
        combined_mask = np.zeros(len(filtered_table), dtype=bool)
        for (x_min, x_max, y_min, y_max) in region_coordinates:
            region_mask = (filtered_table['x'] >= x_min) & (filtered_table['x'] <= x_max) & \
                          (filtered_table['y'] >= y_min) & (filtered_table['y'] <= y_max)
            combined_mask |= region_mask  # Combine masks

        # Apply combined mask to filtered_table
        filtered_table = filtered_table[combined_mask]

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


def find_star_rms(comp_fluxes, airmass):
    comp_star_rms = np.array([])
    Ncomps = comp_fluxes.shape[0]
    for i in range(Ncomps):
        comp_flux = np.copy(comp_fluxes[i])
        airmass_cs = np.polyfit(airmass, comp_flux, 1)
        airmass_mod = np.polyval(airmass_cs, airmass)
        comp_flux_corrected = comp_flux / airmass_mod
        comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
        comp_star_rms_val = np.std(comp_flux_norm)
        if np.isfinite(comp_star_rms_val):
            comp_star_rms = np.append(comp_star_rms, comp_star_rms_val)
        else:
            comp_star_rms = np.append(comp_star_rms, 99.)
    return np.array(comp_star_rms)


def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0, sig_level=3., dmag=0.2):
    comp_star_rms = find_star_rms(comp_fluxes, airmass)
    print(f'RMS of comparison stars: {comp_star_rms}')
    print(f'Number of comparison stars RMS before filtering: {len(comp_star_rms)}')

    # Initialize mask and storage for good and bad stars' data
    comp_star_mask = np.array([True] * len(comp_star_rms))
    cumulative_mask = np.array([True] * len(comp_star_rms))
    i = 0

    # Containers to store final good/bad RMS and magnitudes for plotting
    final_good_rms, final_good_mags = [], []
    final_bad_rms, final_bad_mags = [], []

    while True:
        i += 1
        comp_mags = np.copy(comp_mags0[cumulative_mask])
        comp_rms = np.copy(comp_star_rms[cumulative_mask])
        N1 = len(comp_mags)

        if N1 == 0:
            print("No valid comparison stars left after filtering.")
            break

        edges = np.arange(comp_mags.min(), comp_mags.max() + dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1] + edges[1:]) / 2.

        std_medians = []
        for j in range(1, len(edges)):
            in_bin = comp_rms[dig == j]
            std_medians.append(np.nan if len(in_bin) == 0 else np.median(in_bin))

        std_medians = np.array(std_medians)
        valid_mask = ~np.isnan(std_medians)
        mag_nodes = mag_nodes[valid_mask]
        std_medians = std_medians[valid_mask]

        if len(mag_nodes) < 4:
            if len(mag_nodes) > 1:
                mod = np.interp(comp_mags, mag_nodes, std_medians)
                mod0 = np.interp(comp_mags0, mag_nodes, std_medians)
            else:
                print("Not enough data for interpolation. Skipping iteration.")
                break
        else:
            spl = Spline(mag_nodes, std_medians)
            mod = spl(comp_mags)
            mod0 = spl(comp_mags0)

        std = np.std(comp_rms - mod)
        new_comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        cumulative_mask = cumulative_mask & new_comp_star_mask

        # Store final good/bad data for plotting after the iterations finish
        final_good_rms = comp_star_rms[cumulative_mask]
        final_good_mags = comp_mags0[cumulative_mask]
        final_bad_rms = comp_star_rms[~cumulative_mask]
        final_bad_mags = comp_mags0[~cumulative_mask]

        if N1 == np.sum(cumulative_mask) or i > 10:
            break

    # Plot RMS vs. magnitude for good and bad comparison stars after all iterations
    # Find RMS of the dimmest star among the good stars to set the y-axis limit
    dimmest_good_star_rms = final_good_rms[final_good_mags.argmax()]
    y_limit_high = 2 * dimmest_good_star_rms  # Adjust multiplier as needed for clarity
    y_limit_low = 0.01 * dimmest_good_star_rms  # Adjust multiplier as needed for clarity

    plt.figure()
    plt.scatter(final_good_mags, final_good_rms, color='black')
    plt.scatter(final_bad_mags, final_bad_rms, color='red')
    plt.xlabel('Magnitude')
    plt.ylabel('RMS')
    plt.ylim(y_limit_low, y_limit_high)  # Set y-axis limit based on dimmest good star RMS
    # plt.legend()
    plt.ylim()
    # plt.title('RMS vs. Magnitude of Comparison Stars')
    plt.tight_layout()
    plt.show()

    # save the data for the plot in a json file

    output_data = {
        "good_rms": final_good_rms.tolist(),
        "good_mags": final_good_mags.tolist(),
        "bad_rms": final_bad_rms.tolist(),
        "bad_mags": final_bad_mags.tolist()
    }

    with open(f'rms_vs_mag_comps.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f'RMS of comparison stars after filtering: {len(comp_star_rms[cumulative_mask])}')
    print(f'RMS values after filtering: {comp_star_rms[cumulative_mask]}')

    return cumulative_mask, comp_star_rms, i


def find_best_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size, json):
    # Filter the table based on color/magnitude tolerance
    filtered_table, airmass = limits_for_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size, json)
    tic_ids = np.unique(filtered_table['tic_id'])
    print(f'Number of comparison stars after the filter table in terms of color/mag: {len(tic_ids)}')

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
    print(f'Star with the min rms: {np.min(comp_star_rms)} and tic_id: {tic_ids[np.argmin(comp_star_rms)]}')

    # Filter tic_ids based on the mask
    good_tic_ids = tic_ids[comp_star_mask]

    # Now filter the table based on these tic_ids
    good_comp_star_table = filtered_table[np.isin(filtered_table['tic_id'], good_tic_ids)]

    return good_comp_star_table  # Return the filtered table including only good comp stars


def plot_comp_lc(time_list, flux_list, fluxerr_list, tic_ids, batch_size=9):
    """
    Plot the light curves for comparison stars in batches of `batch_size` (9 per figure by default).
    """
    total_stars = len(tic_ids)
    num_batches = int(np.ceil(total_stars / batch_size))  # Calculate how many batches we need

    for batch_num in range(num_batches):
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Create 3x3 grid of subplots
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i in range(batch_size):
            idx = batch_num * batch_size + i
            if idx >= total_stars:
                break  # Exit if we exceed the number of stars to plot

            tic_id = tic_ids[idx]
            ax = axes[i // 3, i % 3]  # Select the correct subplot

            # Get the current comparison star flux and time
            comp_fluxes = flux_list[idx]
            comp_fluxerrs = fluxerr_list[idx]
            comp_time = time_list[idx]

            # Calculate the sum of all fluxes except the current star's flux
            reference_fluxes_comp = np.sum(np.delete(flux_list, i, axis=0), axis=0)
            reference_fluxerrs_comp = np.sqrt(np.sum(np.delete(fluxerr_list, i, axis=0) ** 2, axis=0))

            # Normalize the current star's flux by the sum of the other comparison stars' fluxes
            comp_fluxes_dt = comp_fluxes / reference_fluxes_comp
            comp_fluxerrs_dt = np.sqrt(comp_fluxerrs ** 2 + reference_fluxerrs_comp ** 2)
            # Normalize the star's flux by the mean flux
            # comp_fluxes_dt = comp_fluxes_dt / np.mean(comp_fluxes_dt)

            # Bin the data (optional, can be skipped if not needed)
            comp_time_dt, comp_fluxes_dt_binned, comp_fluxerrs_dt_binned = (
                bin_time_flux_error(comp_time, comp_fluxes_dt, comp_fluxerrs_dt, 12))

            # Plot the light curve in the current subplot
            ax.plot(comp_time_dt, comp_fluxes_dt_binned, 'o', color='blue', alpha=0.8)
            ax.set_title(f'Comparison star: {tic_id}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')

        plt.tight_layout()
        plt.show()


def get_phot_files(directory):
    """
    Function to retrieve the first photometry file from a given directory.
    Returns the filename as a string.
    """
    files = [f for f in os.listdir(directory) if f.startswith('phot') and f.endswith('.fits')]
    if len(files) == 0:
        raise FileNotFoundError("No FITS files found in the directory.")
    return files  # Return the first FITS file found as a string


def main():
    # Add parse for tic_id_to_plot
    parser = argparse.ArgumentParser(description='Plot light curves for a given TIC ID.')
    parser.add_argument('tic_id', type=int, help='TIC ID to plot the light curve for.')
    parser.add_argument('--aper', type=int, default=5, help='Aperture number to use for photometry.')
    parser.add_argument('--cam', type=str, default='CMOS', help='Aperture number to use for photometry.')
    parser.add_argument('--pos', action='store_true', help='Plot comp stars positions on the image.')
    parser.add_argument('--dmb', type=float, default=0.5, help='Brighter comparison star threshold (default: 0.5 mag)')
    parser.add_argument('--dmf', type=float, default=1.5, help='Fainter comparison star threshold (default: 1.5 mag)')
    parser.add_argument('--crop', type=int, help='Crop size for comparison stars (optional)')
    parser.add_argument('--json_file', action='store_true', help='Use JSON file for region filtering (optional)')
    # Add argument to provide a txt file if comparison stars are known
    parser.add_argument('--comp_stars', type=str, help='Text file with known comparison stars.')

    args = parser.parse_args()
    tic_id_to_plot = args.tic_id
    APERTURE = args.aper
    DM_BRIGHT = args.dmb
    DM_FAINT = args.dmf
    camera = args.cam
    crop_size = args.crop
    fwhm_pos = args.json_file
    current_night_directory = os.getcwd()  # Change this if necessary

    # Read the photometry file
    phot_files = get_phot_files(current_night_directory)
    for phot_file in phot_files:
        print(f'Photometry file: {phot_file}')

        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        # Extract data for the specific TIC ID
        if tic_id_to_plot in np.unique(phot_table['tic_id']):
            print(f"Performing relative photometry for TIC ID = {tic_id_to_plot}")

            if args.comp_stars:
                # Read the file with known comparison stars
                comp_stars_file = args.comp_stars
                comp_stars = np.loadtxt(comp_stars_file, dtype=int)
                # Use the tic_ids directly from the phot_table
                tic_ids = np.intersect1d(comp_stars, np.unique(phot_table['tic_id']))
                print(f'Found {len(tic_ids)} comparison stars from the file.')
            else:
                # Find the best comparison stars
                best_comps_table = find_best_comps(phot_table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT,
                                                   crop_size, fwhm_pos)
                tic_ids = np.unique(best_comps_table['tic_id'])
                print(f'Found {len(tic_ids)} comparison stars from the analysis')

            time_list = []
            flux_list = []
            fluxerr_list = []

            # Collect time, flux, and flux error data
            for tic_id in tic_ids:
                if args.comp_stars:
                    # If comparison stars are loaded from the file, do not call find_best_comps
                    comp_time = phot_table[phot_table['tic_id'] == tic_id]['jd_mid']
                    comp_fluxes = phot_table[phot_table['tic_id'] == tic_id][f'flux_{APERTURE}']
                    comp_fluxerrs = phot_table[phot_table['tic_id'] == tic_id][f'fluxerr_{APERTURE}']
                else:
                    # If no comp_stars file, use best_comps_table
                    comp_time = best_comps_table[best_comps_table['tic_id'] == tic_id]['jd_mid']
                    comp_fluxes = best_comps_table[best_comps_table['tic_id'] == tic_id][f'flux_{APERTURE}']
                    comp_fluxerrs = best_comps_table[best_comps_table['tic_id'] == tic_id][f'fluxerr_{APERTURE}']

                time_list.append(comp_time)
                flux_list.append(comp_fluxes)
                fluxerr_list.append(comp_fluxerrs)

            # Convert lists to arrays
            flux_list = np.array(flux_list)
            fluxerr_list = np.array(fluxerr_list)
            time_list = np.array(time_list)

            # Reference fluxes and errors (sum of all stars, excluding the target star)
            reference_fluxes = np.sum(flux_list, axis=0)
            reference_fluxerrs = np.sqrt(np.sum(fluxerr_list ** 2, axis=0))

            # Bin the master reference data
            time_list_binned, reference_fluxes_binned, reference_fluxerrs_binned = (
                bin_time_flux_error(time_list[0], reference_fluxes, reference_fluxerrs, 12))

            if args.pos:
                # Plot the comparison stars' positions on the image
                plot_comps_position(phot_table, tic_id_to_plot, tic_ids, camera)

            # Call the plot function
            plot_comp_lc(time_list, flux_list, fluxerr_list, tic_ids)

            # Perform relative photometry for target star and plot
            target_star = phot_table[phot_table['tic_id'] == tic_id_to_plot]
            target_flux = target_star[f'flux_{APERTURE}']
            target_fluxerr = target_star[f'fluxerr_{APERTURE}']
            target_time = target_star['jd_mid']

            # Bin the target star data and do the relative photometry
            target_time_binned, target_fluxes_binned, target_fluxerrs_binned = (
                bin_time_flux_error(target_time, target_flux, target_fluxerr, 12))
            flux_ratio_binned = target_fluxes_binned / reference_fluxes_binned
            flux_ratio_mean_binned = np.median(flux_ratio_binned)
            target_fluxes_dt_binned = flux_ratio_binned / flux_ratio_mean_binned
            RMS_binned = np.std(target_fluxes_dt_binned)

            # Calculate the flux ratio for the target star with respect to the summation of the reference stars' fluxes
            flux_ratio = target_flux / reference_fluxes
            flux_err_ratio = np.sqrt((target_fluxerr/target_flux**2) + (reference_fluxerrs/reference_fluxes**2))
            flux_err = flux_ratio * flux_err_ratio

            # Calculate the average flux ratio of the target star
            flux_ratio_mean = np.median(flux_ratio)

            # Normalize the flux ratio (result around unity)
            target_fluxes_dt = flux_ratio / flux_ratio_mean
            target_flux_err_dt = flux_err / flux_ratio_mean

            # Estimate the RMS of the target star
            RMS = np.std(target_fluxes_dt)

            print(f'RMS for Target: {RMS * 100:.3f}% and binned: {RMS_binned * 100:.3f}%')
            plt.plot(target_time_binned, target_fluxes_dt_binned, 'o', color='red', label=f'RMS unbinned = {RMS:.4f}')
            plt.title(f'Target star: {tic_id_to_plot}, Tmag = {target_star["Tmag"][0]}')
            plt.legend(loc='best')
            plt.show()

            # Save target_time_binned and target_fluxes_dt in a JSON file
            data_to_save = {
                "TIC_ID": tic_id_to_plot,
                "Time_BJD": target_time.tolist(),
                "Relative_Flux": target_fluxes_dt.tolist(),
                "Relative_Flux_err": target_flux_err_dt.tolist(),
                "RMS": RMS.tolist()
            }

            json_filename = f'target_light_curve_{tic_id_to_plot}_{camera}.json'
            with open(json_filename, 'w') as json_file:
                json.dump(data_to_save, json_file, indent=4)

            print(f'Data saved to {json_filename}')

            # Save tic_ids used for comparison stars in a txt file
            comp_stars_filename = f'comp_stars_{tic_id_to_plot}_{camera}.txt'

            with open(comp_stars_filename, 'w') as comp_stars_file:
                for tic_id in tic_ids:
                    comp_stars_file.write(f'{tic_id}\n')
        else:
            print(f"No data found for TIC ID = {tic_id_to_plot}")


if __name__ == "__main__":
    main()
