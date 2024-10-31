#!/bin/bash

# Record the start time
start_time=$(date +%s)

echo "Starting processing..."

# Create directories.json with the specified paths
cat <<EOF > directories.json
{
  "calibration_paths": [
    "/Users/u5500483/Downloads/DATA_MAC/CMOS/20231212/",
    "/home/ops/data/20231212/"
  ],
  "base_paths": [
    "/Users/u5500483/Downloads/DATA_MAC/CMOS/",
    "/home/ops/data/"
  ],
  "out_paths": [
    "/Users/u5500483/Downloads/DATA_MAC/CMOS/calibration_images/",
    "/home/ops/data/calibration_images/"
  ]
}
EOF

# Run the Python scripts
python /home/ops/refcatpipe2/cmos/simple_wrapper.py
python /home/ops/ngcmos/check_cmos.py
python /home/ops/ngcmos/adding_headers.py
python /home/ops/ngcmos/create_flats.py
python /home/ops/ngcmos/process_cmos.py
python /home/ops/fwhm_stars/fhwm_batches.py --size 11 # make plot and save to fwhm_results.json
python /home/ops/ngcmos/relative_phot_dev.py --aper 5
python /home/ops/fwhm_stars/best_fwhm.py --size 11 # save to fwhm_positions.json
python /home/ops/ngcmos/remove_fits_files.py

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"