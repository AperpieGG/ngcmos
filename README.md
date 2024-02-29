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

The pipeline peforms photometry with a CMOS camera at NGTS

The scripts are running under the process_cmos.py. This is a modified code from process_leowasp.py (James McCormac all right reserved).

Main rutines for files selection and management are running on that code.

Info and utilities are taken from utils.py. 

Reduction of the images from calibration_images.py

The process_cmos.py works:

1) Run it from the data directory
2) It uses config files to identify base_path and out_paths
3) Function to find the current night directory based on the previous night of observation. If it doesnt exist it will use the directory the script is running.
4) Function to filter the filenames. It will exclude files such flats, bias, darks, caralogs and photometry.
5) Function to get prefix. This basically finds the NGTS field that were observed on the night.
6) Main function that uses the catalog file for each image after reduction and extracts the photometry for all stars on the field. It saves them a fits file.


The analyse_cmos.py works:

1) 

