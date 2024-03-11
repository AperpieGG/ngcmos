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

## Run the following
The script ran as cronjobs in the nuc computer which is in Chile currently. The path will be found from the directories.json files that exist in to the directory where the data lies.


1) ```QC_Donuts.py ```
   This is a script to run Donuts on a specified observing night, compute and plots the x-y pixels shifts, and save the results to a JSON file.
   It also created an mp4 animation of the images with shifts greater than 0.5 pixels.
3) ```QC_control.py ```
   This is script that will create an mp4 video animation for each fifth image of each object of each particular observing night.
4) ```calibration_images.py```

5) ```Simple_wrapper.py ```
   This script is made of two individual scripts that will solve the images astrometrically from the tic8 catalog and also tweak the solution for the distortion effect. Finally, a master catalog and an input catalog for the stars on the field are created. The input catalog has specific filters that will cut the candidates down and will be used for photometry later.
6) ```check_cmos.py```
   After the wcs is updated on the headers, this script will check which images will be used for photometry. First uses donuts to measure the shifts and then it will check if the wcs is on the headers. If not, the data will be moved to the no_wcs/ subdirectory
7) ```adding_headers.py```
   This code is generating the Airmass and Filter keyword in the headers of the data
8) process_cmos.py
9) analyse_cmos.py

The process_cmos.py works:

1) Run it from the data directory
2) It uses config files to identify base_path and out_paths
3) Function to find the current night directory based on the previous night of observation. If it doesnt exist it will use the directory the script is running.
4) Function to filter the filenames. It will exclude files such flats, bias, darks, caralogs and photometry.
5) Function to get prefix. This basically finds the NGTS field that were observed on the night.
6) Main function that uses the catalog file for each image after reduction and extracts the photometry for all stars on the field. It saves them a fits file.


The analyse_cmos.py works:

1) 

