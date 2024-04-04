#!/bin/bash

echo "Starting processing..."

# Run the Python scripts
python /home/ops/refcatpipe2/cmos/simple_wrapper_ccd.py
python /home/ops/ngcmos/check_ccd.py
python /home/ops/ngcmos/process_ccd.py

echo "Finishing processing!"