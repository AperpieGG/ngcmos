#!/bin/bash

# Record the start time
start_time=$(date +%s)

echo "Starting processing..."
#
## Run the initial Python scripts
#python /home/ops/ngcmos/unzip_fits.py     # unzips the FITS files and deletes the bz2 extension
#python /home/ops/ngcmos/trim_ccd.py
#python /home/ops/refcatpipe2/cmos/simple_wrapper_ccd.py
#python /home/ops/ngcmos/check_ccd.py
#python /home/ops/ngcmos/calibration_images_ccd.py
#python /home/ops/ngcmos/process_ccd.py

# Check for a subdirectory that matches "action*observeField"
observe_dir=$(find . -maxdepth 1 -type d -name "action*_observeField" | head -n 1)

if [[ -d "$observe_dir" ]]; then
    echo "Found observeField directory: $observe_dir"
    # shellcheck disable=SC2164
    cd "$observe_dir"
    echo "Changed to directory: $(pwd)"

    # Run the additional Python scripts within the observeField subdirectory
    python /home/ops/fwhm_stars/fwhm_batches.py --size 13.5 --cam CCD   # make plot and save to fwhm_results.json
    python /home/ops/ngcmos/relative_phot_dev.py --aper 4
    python /home/ops/fwhm_stars/best_fwhm.py --size 13.5                # save to fwhm_positions.json
    python /home/ops/ngcmos/zip_fits.py                                 # zip the FITS files to bz2 and delete .fits files
else
    echo "No matching observeField directory found."
fi

# Return to the original directory
# shellcheck disable=SC2164
cd -

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"