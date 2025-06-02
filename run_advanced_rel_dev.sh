#!/bin/bash

# Path to your .phot file
PHOT_FILE="phot_NG2320-1302.fits"


# Check if 'targets' directory exists
if [ -d "targets" ]; then
  echo "📁 'targets/' directory already exists. Skipping optimization and file move."
  echo "Running plot_timescale_json.py..."
  python3 plot_timescale_json.py
  exit 0
fi

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

# Now parse and execute each line from best_params_log.txt
LOG_FILE="best_params_log.txt"

if [[ -f "$LOG_FILE" ]]; then
  echo "Executing best parameter configurations from $LOG_FILE..."

  while IFS= read -r line; do
    # Strip everything after the first '#' (the comment)
    cmd=$(echo "$line" | cut -d'#' -f1)

    # Run the command with Python
    echo "Executing: $cmd"
    python3 -c "import sys; sys.argv = ['$cmd']; exec(open('rel_dev_dev.py').read())"
  done < "$LOG_FILE"
else
  echo "No best_params_log.txt found."
fi


# Create the 'targets' directory if it doesn't exist
mkdir -p targets

# Move all target light curve JSON files into the 'targets' folder
mv target_light_curve*.json targets/

echo "Moved all target_light_curve JSON files to ./targets/"

# Run the plot_timescale_json.py script
python3 /home/ops/ngcmos/plot_timescale_json.py

echo "✅ Finished running plot_timescale_json.py"