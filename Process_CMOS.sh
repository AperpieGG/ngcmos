#!/bin/bash

echo "Starting processing..."

# Run the Python scripts
python /home/ops/refcatpipe2/simple_wrapper.py
python /home/ops/ngcmos/check_cmos.py
python /home/ops/ngcmos/adding_headers.py
python /home/ops/ngcmos/process_cmos.py

echo "Finishing processing!"