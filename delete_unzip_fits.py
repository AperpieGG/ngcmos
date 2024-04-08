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


if __name__ == "__main__":
    delete_images()
