# ngcmos
Rep to do photometry from ngts rep 
Rep to do simulations before real data

## Installation
```bash
git clone
cd ngcmos
pip install -r requirements.txt
```

## Description

The pipeline peforms photometry from with the CMOS at NGTS

The scripts are running under the process_cmos.py. 

Main rutines for files selection and management are running on that code.

Info and utilities are taken from utils.py. 

Reduction of the images from calibration_images.py

The process_cmos.py works:

1) Run it from the data directory
2) It uses config files to identify base_path and out_paths:

