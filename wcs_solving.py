import os
from datetime import datetime, timedelta
from astropy.io import fits
import shutil


def solve_reference_image(refimage, backup_folder):
    """
    Solve the reference image to obtain better WCS

    Parameters
    ----------
    refimage : string
        path to the reference image for solving
    backup_folder : string
        path to the folder to store backup files

    Returns
    -------
    solved : boolean
        Did the image solve ok?

    Raises
    ------
    None
    """
    solved_refimage = f"{os.path.splitext(refimage)[0]}.new"

    if not os.path.exists(solved_refimage):
        with fits.open(refimage) as hdulist:
            header = hdulist[0].header
            ra = float(header['TELRAD'])
            dec = float(header['TELDECD'])

        # Build the astrometry.net command
        command = (
            f"solve-field {refimage} "
            f"--ra {ra:.6f} --dec {dec:.6f} "
            "--radius 5 "
            "--overwrite "
            "--scale-units arcsecperpix \
             --scale-low 3.9 --scale-high 4.3"
            "--skip-solved --no-plots --no-verify -z 2 --tweak-order 3 --cpulimit 600"
        )

        try:
            os.system(command)
        except Exception as e:
            print(f"Error solving image: {e}")
            return False

        # Backup the original .fits file
        try:
            shutil.move(refimage, os.path.join(backup_folder, os.path.basename(refimage)))
            print(f"Moved original file to backup folder: {os.path.basename(refimage)}")
        except Exception as e:
            print(f"Error moving original file to backup folder: {e}")

        # Rename the solved image to use .fits instead of .new
        os.rename(solved_refimage, f"{os.path.splitext(refimage)[0]}.fits")
        print(f"Renamed solved image to: {os.path.splitext(refimage)[0]}.fits")

    return os.path.exists(f"{os.path.splitext(refimage)[0]}.fits")


def solve_all_images_in_directory(directory, backup_folder):
    """
    Solve WCS for all FITS images in the specified directory

    Parameters
    ----------
    directory : string
        path to the directory containing FITS images

    Returns
    -------
    None
    """
    exclude_words = ["evening", "morning", "flat", "bias", "dark"]

    # Get a sorted list of FITS files in the directory
    fits_files = sorted([filename for filename in os.listdir(directory) if filename.endswith(".fits")
                         and not any(word in filename.lower() for word in exclude_words)])

    for filename in fits_files:
        filepath = os.path.join(directory, filename)
        if solve_reference_image(filepath, backup_folder):
            print(f"WCS solved successfully for {filename}")
        else:
            print(f"Failed to solve WCS for {filename}")


def remove_unwanted_files(directory):
    unwanted_extensions = ['.xyls', '.axy', '.corr', '.match', '.rdls', '.solved', '.wcs']

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in unwanted_extensions):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Removed unwanted file: {filename}")
            except Exception as e:
                print(f"Error removing file {filename}: {e}")


def find_current_night_directory(file_path):
    # Get the current date in the format YYYYMMDD
    current_date = datetime.now().strftime("%Y%m%d") + '/'
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d") + '/'

    # Construct the path for the previous_date directory
    current_date_directory = os.path.join(file_path, previous_date)

    # Check if the directory exists
    if os.path.isdir(current_date_directory):
        return current_date_directory
    else:
        return None


def zip_backup_folder(backup_folder):
    # Zip the contents of the backup folder directly
    zip_filename = f"{os.path.splitext(backup_folder)[0]}"
    shutil.make_archive(zip_filename, 'zip', os.path.dirname(backup_folder), os.path.basename(backup_folder))
    print(f"Backup folder zipped successfully: {zip_filename}")


def main():
    file_path = "/Users/u5500483/Downloads/DATA_MAC/CMOS/"
    current_night_directory = find_current_night_directory(file_path)

    # Rest of the code remains the same
    if current_night_directory:
        print(f"Current night directory found: {current_night_directory}")

        # Directly create the backup zip file without creating a backup folder
        previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        backup_folder = os.path.join(current_night_directory, f"backup_{previous_date}")
        os.mkdir(backup_folder)
        print(f"Backup folder created: {backup_folder}")
        # Proceed with solving and other operations
        solve_all_images_in_directory(current_night_directory, backup_folder)
        remove_unwanted_files(current_night_directory)

        # Zip the backup folder
        zip_backup_folder(backup_folder)
        shutil.rmtree(backup_folder)
        print(f"Backup folder zipped successfully: {backup_folder}")

    else:
        print("No current night directory found.")


if __name__ == "__main__":
    main()
