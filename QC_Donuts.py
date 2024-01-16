#!/Users/u5500483/anaconda3/bin/python
import json
import os
from datetime import datetime
from donuts import Donuts
import glob
from matplotlib import pyplot as plt
import warnings
from astropy.io import fits
from astropy.time import Time


warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.fromnumeric")
warnings.filterwarnings("ignore", category=UserWarning, module="donuts.image")


def plot_images():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


def find_current_night_directory(base_path):
    # Get the current date in the format YYYYMMDD
    current_date = datetime.now().strftime("%Y%m%d")

    # Construct the path for the current date directory
    current_date_directory = os.path.join(base_path, current_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return None


def find_first_image_of_each_prefix(directory):
    # List all items (files and directories) in the given directory
    items = os.listdir(directory)

    # Filter out files with the word "flat, dark, bias" in their names
    filtered_items = [item for item in items if "flat" and "bias" and "dark" not in item.lower()]

    # Dictionary to store the first image of each prefix
    first_image_of_each_prefix = {}

    # Iterate through filtered items
    for item in filtered_items:
        # Extract the first 6 letters of the item
        prefix = item[:11]

        # Check if the prefix is already a key in the dictionary
        if prefix not in first_image_of_each_prefix:
            # Update the file path pattern for the given prefix
            pattern = os.path.join(directory, f'{prefix}*.fits')
            # Use glob to find matching files
            matching_files = glob.glob(pattern)
            # Sort the matching files
            matching_files = sorted(matching_files)
            # Check if any matching files were found
            if matching_files:
                first_image_of_each_prefix[prefix] = matching_files[0]

    # Print the first image for each different prefix
    print(f"First image of each prefix in {directory} (excluding those with 'flat', 'bias' and 'dark' in the name):\n")
    for prefix, first_image in first_image_of_each_prefix.items():
        print(f"Prefix: {prefix}, First Image: {first_image}")
        run_donuts(directory, prefix)


def run_donuts(directory, prefix):
    path = directory + '/'
    image_names = glob.glob(path + f'{prefix}*.fits')
    image_names = sorted(image_names)

    if not image_names:
        print(f"No images found for prefix: {prefix}")
        return

    reference_image_name = image_names[0]

    # Print some debugging information
    print(f"Using {reference_image_name} as the reference image for prefix: {prefix}\n")

    science_image_names = [f for f in glob.glob(path + f'{prefix}*.fits')[1:]]
    science_image_names = sorted(science_image_names)

    d = Donuts(
        refimage=reference_image_name,
        image_ext=0,
        overscan_width=20,
        prescan_width=20,
        border=64,
        normalise=True,
        exposure='EXPTIME',
        subtract_bkg=True,
        ntiles=32)
    # for each image, compute the x/y translation required
    # to align the images onto the reference image

    x_shifts = []
    y_shifts = []

    for image in science_image_names:
        shift_result = d.measure_shift(image)
        x = shift_result.x
        y = shift_result.y

        if abs(x.value) < 0.5 and abs(y.value) < 0.5:
            print("Image {} with shifts (x, y): {}, {}".format(image[-22:], x.value, y.value))
        elif abs(x.value) >= 0.5 or abs(y.value) >= 0.5:
            print('WARNING: Image {} is not aligned with shifts (x, y): {}, {}'.format(image[-22:], x.value, y.value))
        else:
            pass

        # Append shift values to the lists
        x_shifts.append(x.value)
        y_shifts.append(y.value)

    num_large_shifts = sum(1 for x, y in zip(x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5)
    print("The number of images with shifts greater than 0.5 pixels is: {}".format(num_large_shifts))
    print()
    save_results(x_shifts, y_shifts, reference_image_name, save_path, prefix, science_image_names)

    time = acquire_header_info(directory, prefix)

    plot_shifts(x_shifts, y_shifts, save_path, prefix, time)


def utc_to_jd(utc_time_str):
    # Convert DATE-OBS to Astropy Time object
    t = Time(utc_time_str, format='isot', scale='utc')

    # Convert to Julian Date
    jd = t.jd

    return jd


def acquire_header_info(directory, prefix):
    path = directory + '/'
    image_names = glob.glob(path + f'{prefix}*.fits')
    image_names = sorted(image_names[1:])
    time_jd = []

    for image in image_names:
        with fits.open(image) as hdulist:
            header = hdulist[0].header
            # Extract the UTC time string from the header
            utc_time_str = header['DATE-OBS']

            # Convert the UTC time string to JD
            jd = utc_to_jd(utc_time_str)

            time_jd.append(jd)
            print(utc_time_str)
    return time_jd


def plot_shifts(x_shifts, y_shifts, save_path, prefix, time):
    # Plot the shifts with colorbar
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x_shifts, y_shifts, c=time, cmap='viridis', label='Shifts for field: {}'.format(prefix), marker='o')
    plt.xlabel('X Shift (pixels)')
    plt.ylabel('Y Shift (pixels)')
    plt.title('Shifts with respect to the ref image')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)  # Add horizontal line at y=0
    plt.axvline(0, color='black', linestyle='-', linewidth=1)  # Add vertical line at x=0
    plt.legend()

    # Set the axes limits to center (0, 0)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Add colorbar
    cbar = plt.colorbar(scatter, label='Time')

    # Get the current date in the format DDMMYYYY
    timestamp = datetime.now().strftime("%Y%m%d")

    # Construct the directory path based on the current date
    base_file_name = f"donuts_{prefix}_{timestamp}"

    # Construct the full file path within the "shifts_plots" directory
    pdf_file_path = os.path.join(save_path, f"{base_file_name}.pdf")

    # Save the figure
    fig.savefig(pdf_file_path, bbox_inches='tight')
    print(f"PDF plot saved to: {pdf_file_path}\n")


def save_results(x_shifts, y_shifts, reference_image_name, save_path, prefix, science_image_names):
    # Create a timestamp for the file name
    timestamp = datetime.now().strftime("%Y%m%d")

    # Construct the base file name
    base_file_name = f"donuts_{prefix}_{timestamp}"

    # Construct the full file paths
    json_file_path = os.path.join(save_path, f"{base_file_name}.json")

    num_large_shifts = sum(1 for x, y in zip(x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5)

    # Save the results to the JSON file
    results_data = {
        "Reference Image": reference_image_name,
        "The number of images with shifts greater than +/-0.5 pixels is": {
            "Total Images": len(science_image_names),
            "Number of Images with Large Shifts": num_large_shifts
        },
        "The name of the images with shifts greater than 0.5 pixels is":
            [image for image, x, y in zip(science_image_names, x_shifts, y_shifts) if abs(x) >= 0.5 or abs(y) >= 0.5],
        "X Shifts and Y Shifts": list(zip(x_shifts, y_shifts)),
    }

    with open(json_file_path, 'w') as json_file:
        json.dump(results_data, json_file, indent=4)

    print(f"JSON results saved to: {json_file_path}")


if __name__ == "__main__":
    # Specify the base path
    base_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/'
    save_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/shifts_plots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Find the current night directory
    current_night_directory = find_current_night_directory(base_path)

    if current_night_directory:
        print(f"Current night directory found: {current_night_directory}")
        find_first_image_of_each_prefix(current_night_directory)
    else:
        print("No current night directory found.")







