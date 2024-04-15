#!/bin/bash

echo "Starting processing..."

# Run the Python scripts
python /home/ops/ngcmos/unzip_fits.py # unzipped the fits files and delete the bz2 extension
python /home/ops/ngcmos/trim_ccd.py
python /home/ops/refcatpipe2/cmos/simple_wrapper_ccd.py
python /home/ops/ngcmos/check_ccd.py
python /home/ops/ngcmos/calibration_images_ccd.py
python /home/ops/ngcmos/process_ccd.py
python /home/ops/ngcmos/zip_fits.py # zipped the fits to bz2 and deleted .fits files

echo "Finishing processing!"