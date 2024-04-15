#! /usr/bin/env python
import os
import bz2


def compress_fits_files(directory):
    # Get a list of all .fits files in the directory
    fits_files = [f for f in os.listdir(directory) if f.startswith('IMAGE') and f.endswith('.fits')]

    # Compress each .fits file into .fits.bz2 format
    for fits_file in fits_files:
        fits_path = os.path.join(directory, fits_file)
        fits_bz2_path = fits_path + '.bz2'
        with open(fits_path, 'rb') as source:
            with bz2.BZ2File(fits_bz2_path, 'wb') as target:
                target.write(source.read())
        print(f"{fits_file} compressed to {fits_bz2_path}")


def delete_uncompressed_images(directory):
    # Get a list of all uncompressed .fits files in the directory
    uncompressed_files = [f for f in os.listdir(directory) if f.startswith('IMAGE') and f.endswith('.fits')]

    # Delete each uncompressed .fits file
    for file in uncompressed_files:
        os.remove(os.path.join(directory, file))
        print(f"{file} deleted.")


def main():
    # Get the current working directory
    parent_directory = os.getcwd()

    # Get a list of subdirectories inside the parent directory
    subdirectories = [name for name in os.listdir(parent_directory) if
                      os.path.isdir(os.path.join(parent_directory, name))]

    print('The subdirectories are:', subdirectories)

    # Iterate over each subdirectory
    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_biasFrames"):
            # Form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # Change the working directory to the subdirectory
            os.chdir(subdirectory_path)

            print(f"Processing subdirectory: {subdirectory_path}")
            compress_fits_files(subdirectory_path)
            delete_uncompressed_images(subdirectory_path)


if __name__ == "__main__":
    main()
