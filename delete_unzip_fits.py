#!/usr/bin/env python3

import os


def delete_images():
    image_directory = os.getcwd()

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.startswith("IMAGE") and f.endswith(".fits")]

    # Exclude image files that end with ".fits.bz2"
    image_files = [f for f in image_files if not f.endswith(".fits.bz2")]

    # Delete the selected image files
    for file in image_files:
        os.remove(os.path.join(image_directory, file))
        print(f"Deleted: {file}")

    print("Deletion of images has been completed.")


def main():
    # Get the current working directory
    parent_directory = os.getcwd()

    # Get a list of subdirectories inside the parent directory
    subdirectories = [name for name in os.listdir(parent_directory) if
                      os.path.isdir(os.path.join(parent_directory, name))]

    print('The subdirectories are:', subdirectories)

    # Iterate over each subdirectory
    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_observeField"):
            # Form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # Change the working directory to the subdirectory
            os.chdir(subdirectory_path)

            print(f"Processing subdirectory: {subdirectory_path}")
            delete_images()


if __name__ == "__main__":
    main()
