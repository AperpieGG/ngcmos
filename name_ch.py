#!/usr/bin/env python
import os

# Directory containing the images
directory = os.getcwd()

# List all files in the directory
files = os.listdir(directory)

# Iterate over each file
for filename in files:
    # Check if the file starts with 'stacked_' and ends with '.fits'
    if filename.startswith('reduced_') and filename.endswith('.fits'):
        # Extract the part of the filename after 'stacked_'
        new_filename = filename[len('reduced_'):]
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        print(f"Renamed {filename} to {new_filename}")