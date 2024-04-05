#!/bin/bash

echo "Starting processing..."

# Run the Python scripts
python /home/ops/ngcmos/unzip_fits.py
python /home/ops/ngcmos/simple_wrapper_ccd.py
python /home/ops/ngcmos/check_ccd.py
python /home/ops/ngcmos/process_ccd.py

echo "Finishing processing!"