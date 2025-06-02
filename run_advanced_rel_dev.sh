#!/bin/bash

# Path to your .phot file
PHOT_FILE="phot_NG2320-1302.fits"

# Extract unique TIC IDs with Tmag between 10 and 11 using Python
output=$(python3 - <<END
from astropy.io import fits
import numpy as np

with fits.open("$PHOT_FILE") as hdul:
    data = hdul[1].data
    tic_ids = data['TIC_ID']
    tmags = data['Tmag']

# Filter by Tmag
mask = (tmags >= 10) & (tmags <= 11)
unique_ids = np.unique(tic_ids[mask])

# Print unique TIC IDs (space-separated) and count, separated by |
print(" ".join(str(tic) for tic in unique_ids) + "|" + str(len(unique_ids)))
END
)

# Separate the output into TIC IDs and count
IFS='|' read -r tic_ids count <<< "$output"

# Print the count
echo "Found $count unique TIC IDs with Tmag between 10 and 11"

# Loop through each TIC ID and run the optimization script
for tic_id in $tic_ids; do
  echo "Running optimization for TIC $tic_id"
  python3 /home/ops/ngcmos/advanced_rel_dev.py --tic_id "$tic_id"
done