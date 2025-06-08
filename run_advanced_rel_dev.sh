#!/bin/bash

# Path to your .phot file
PHOT_FILE="phot_NG2320-1302.fits"

# shellcheck disable=SC2034
TMAG_Bright=$1
# shellcheck disable=SC2034
TMAG_Faint=$2

cam=$3

# Check if 'targets' directory exists
if [ -d "targets" ]; then
  echo "📁 'targets/' directory already exists. Skipping optimization and file move."
  echo "Running plot_timescale_json.py..."
  python3 /home/ops/ngcmos/plot_timescale_json.py
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
mask = (tmags >= $TMAG_Bright) & (tmags <= $TMAG_Faint)
unique_ids = np.unique(tic_ids[mask])

# Take first 45
selected_ids = unique_ids[:45]

# Print selected TICs, selected count, and full count
print(" ".join(str(tic) for tic in selected_ids) + "|" + str(len(selected_ids)) + "|" + str(len(unique_ids)))
END
)

# Separate the output into TIC IDs, selected count, and full count
IFS='|' read -r tic_ids selected_count full_count <<< "$output"

# Print the count
echo "Selected $selected_count out of $full_count TIC IDs with Tmag between $TMAG_Bright and $TMAG_Faint"

# Loop through each TIC ID and run the optimization script
for tic_id in $tic_ids; do
  echo "Running optimization for TIC $tic_id"
  python3 /home/ops/ngcmos/advanced_rel_dev.py --tic_id "$tic_id" --cam "$cam"
done

# Now parse and execute each line from best_params_log.txt
LOG_FILE="best_params_log.txt"

if [[ -f "$LOG_FILE" ]]; then
  echo "Executing best parameter configurations from $LOG_FILE..."
  # Read the log file line by line
  while IFS= read -r line; do
    # Remove comment and trim whitespace
    cmd=$(echo "$line" | cut -d'#' -f1 | xargs)

    if [[ -n "$cmd" ]]; then
      echo "Executing: $cmd"
      eval "$cmd" || echo "⚠️ Failed to execute: $cmd"
    fi
  done < "$LOG_FILE"
else
  echo "❌ No $LOG_FILE found."
fi


# Create the 'targets' directory if it doesn't exist
mkdir -p targets

# Move all target light curve JSON files into the 'targets' folder
mv target_light_curve*.json targets/

echo "Moved all target_light_curve JSON files to ./targets/"

# Run the plot_timescale_json.py script
python3 /home/ops/ngcmos/plot_timescale_json.py

echo "✅ Finished running plot_timescale_json.py"