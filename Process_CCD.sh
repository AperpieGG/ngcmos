#!/bin/bash

echo "Starting processing..."

# Run the Python scripts
python /home/ops/ngcmos/unzip_fits.py
python /home/ops/ngcmos/trim_ccd.py
python /home/ops/refcatpipe2/cmos/simple_wrapper_ccd.py
python /home/ops/ngcmos/check_ccd.py
python /home/ops/ngcmos/calibration_images_ccd.py
python /home/ops/ngcmos/process_ccd.py
#python /home/ops/ngcmos/delete_unzip_fits.py

echo "Finishing processing!"