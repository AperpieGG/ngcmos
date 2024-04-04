#! /usr/bin/env python
import os
import bz2


def unzip_fits_bz2_files():
    # Get the current working directory
    current_directory = os.getcwd()

    # Get a list of all files in the directory
    files = os.listdir(current_directory)

    # Filter out only the .fits.bz2 files
    fits_bz2_files = [f for f in files if f.endswith('.fits.bz2')]

    # Iterate over each .fits.bz2 file
    for fits_bz2_file in fits_bz2_files:
        print(f"Unzipping {fits_bz2_file}...")
        # Form the full path to the .fits.bz2 file
        fits_bz2_path = os.path.join(current_directory, fits_bz2_file)
        # Open the .bz2 file
        with bz2.BZ2File(fits_bz2_path, 'rb') as source:
            # Read the content
            uncompressed_content = source.read()
            # Remove the .bz2 extension to get the output filename
            output_filename = fits_bz2_file[:-4]
            # Form the full path to the output file
            output_path = os.path.join(current_directory, output_filename)
            # Write the uncompressed content to the output file
            with open(output_path, 'wb') as target:
                target.write(uncompressed_content)
        print(f"{fits_bz2_file} unzipped to {output_path}")

    print("Unzipping complete.")


if __name__ == "__main__":
    unzip_fits_bz2_files()
