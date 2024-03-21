#!/bin/bash

echo "Starting processing..."

# Run the Python scripts
python /home/ops/ngcmos/QC_Donuts.py
python /home/ops/ngcmos/QC_control.py
python /home/ops/ngcmos/calibration_images.py
python /home/ops/ngcmos/Simple_wrapper.py
python /home/ops/ngcmos/check_cmos.py
python /home/ops/ngcmos/adding_headers.py
python /home/ops/ngcmos/process_cmos.py

echo "Finishing processing!"