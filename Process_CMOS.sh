#!/bin/bash

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