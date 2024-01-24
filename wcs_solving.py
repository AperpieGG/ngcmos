import os
from astropy.io import fits


def solve_reference_image(refimage):
    """
    Solve the reference image to obtain better WCS

    Parameters
    ----------
    refimage : string
        path to the reference image for solving

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
            f"/opt/homebrew/bin/solve-field {refimage} "
            f"--ra {ra:.6f} --dec {dec:.6f} "
            "--radius 5"
            "--overwrite"
            "--no-plots --overwrite --no-verify -z 2 --tweak-order 3 --cpulimit 600"
        )

        try:
            os.system(command)
        except Exception as e:
            print(f"Error solving image: {e}")
            return False

    return os.path.exists(solved_refimage)


def solve_all_images_in_directory(directory):
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
    for filename in os.listdir(directory):
        if filename.endswith(".fits"):
            filepath = os.path.join(directory, filename)
            if solve_reference_image(filepath):
                print(f"WCS solved successfully for {filename}")
            else:
                print(f"Failed to solve WCS for {filename}")


# Example usage:
directory_path = "/Users/u5500483/Downloads/DATA_MAC/CMOS/testing/"
solve_all_images_in_directory(directory_path)



