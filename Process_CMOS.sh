#!/bin/bash

# Record the start time
start_time=$(date +%s)

echo "Starting processing..."

# Run the Python scripts
python /home/ops/refcatpipe2/cmos/simple_wrapper.py
python /home/ops/ngcmos/check_cmos.py
python /home/ops/ngcmos/adding_headers.py
python /home/ops/ngcmos/create_flats.py
python /home/ops/ngcmos/process_cmos.py
python /home/ops/ngcmos/relative_phot.py
python /home/ops/ngcmos/remove_fits_files.py

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"