#!/bin/bash

echo "Starting processing..."

# Run the Python scripts
python QC_Donuts.py
python QC_control.py
python calibration_images.py
python Simple_wrapper.py
python check_cmos.py
python adding_headers.py
python process_cmos.py

echo "Finishing processing!"