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
   

5) ```simple_wrapper.py ```
   This script is made of two individual scripts that will solve the images astrometrically from the tic8 catalog and also tweak the solution for the distortion effect. Finally, a master catalog and an input catalog for the stars on the field are created. The input catalog has specific filters that will cut the candidates down and will be used for photometry later.
   
6) ```check_cmos.py```
   After the wcs is updated on the headers, this script will check which images will be used for photometry. First uses donuts to measure the shifts and then it will check if the wcs is on the headers. If not, the data will be moved to the no_wcs/ subdirectory.
   
7) ```adding_headers.py```
   This code is generating the Airmass (calculation simple by the unity over the sine of altitude) and Filter keyword in the headers of the data.
   
8) ```process_cmos.py```
   This script rujns from the data directory. It uses the config file to identify the base and out paths. It has a function to find the current night directory based on the previous night of observation. If it doesnt exist it will use the directory the script is running. It will also filter the filenames keeping only the .fits for TOI, excluding flats, bias, darks, caralogs and photometry. To do that it will use the prefix to identify the NGTS fields. The main function uses the catalog file for each image after reduction and extracts the photometry for all stars on the field. It saves them a fits file.
   
   ```calibration_images.py```
   Will search for master bias and darks on the calibration directory. We dont have shutter so bias and darks kept fixed assuming their performance is stable through months of             observations. There are additional steps of how the master_flat is created:
      3. The user, on a given directory has to run the run create_flats.py to create a master_flat.fits. This will be saved on the cwd.
      4. If the user havent run the create_flats.py then as last resort it will use a general master_flat from the calibration path.
         
10) ```analyse_cmos.py```
   This script will do an inspection of the lightcurves using the gaia_id as an argument. It will plot fluc, sky-background and the ROI of the star from the first frame, with the aperture and annulus used.
    



