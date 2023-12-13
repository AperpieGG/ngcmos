#!/Users/u5500483/anaconda3/bin/python
import os
import glob
import warnings
from astropy.io import fits
from astropy import units as u
import numpy as np
import logging
from ccdproc import (
    CCDData,
    combine,
    subtract_bias,
    subtract_dark, flat_correct,
)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger('matplotlib')
logger.handlers = []
logger.propagate = False


class BiasDarkFlatCorrection:
    def __init__(self, path_1, path_2, path_3, path_4):
        self.path_1 = path_1
        self.path_2 = path_2
        self.path_3 = path_3
        self.path_4 = path_4
        self.master_bias = None
        self.master_dark = None
        self.master_flat = None

    def create_master_bias(self):
        # Create a list of the bias images
        bias_list = [f for f in glob.glob(self.path_1 + '*.fits') if 'bias' in fits.getheader(f)['IMAGETYP']]

        ccd_list = []
        for f in bias_list:
            ccd = CCDData.read(f, unit='adu')
            ccd_list.append(ccd)

        # Combine the bias images
        self.master_bias = combine(ccd_list, method='median')
        self.master_bias.write(self.path_2 + 'master_bias.fits', overwrite=True)

    logging.debug('Master bias created')

    def create_master_dark(self, dark_exposure):
        # Create a list of the dark images
        dark_list = [f for f in glob.glob(self.path_1 + '*.fits') if 'dark' in fits.getheader(f)['IMAGETYP']]

        ccd_list = []
        for f in dark_list:
            ccd = CCDData.read(f, unit='adu')
            ccd = subtract_bias(ccd, self.master_bias)
            ccd_list.append(ccd)

        # Combine the dark images
        self.master_dark = combine(ccd_list, method='median')
        self.master_dark.write(self.path_2 + 'master_dark.fits', overwrite=True)

    logging.debug('Master dark created')

    def create_master_flat(self):
        # Create a list of the flat images
        flat_list = [f for f in glob.glob(self.path_1 + '*.fits') if 'flat' in fits.getheader(f)['IMAGETYP']]
        flat_list.sort()

        for f in flat_list:
            with fits.open(f) as fitsfile:
                data_exp = int(fitsfile[0].header[exptime_keyword])
            ccd = CCDData.read(f, unit=u.adu)
            ccd = subtract_bias(ccd, self.master_bias)
            ccd = subtract_dark(ccd, self.master_dark,
                                scale=True,
                                dark_exposure=dark_exposure * u.second,
                                data_exposure=data_exp * u.second)
            ccd.data = ccd.data / np.ma.average(ccd.data)
            flat_list.append(ccd)

        # Combine the flat images
        self.master_flat = combine(flat_list, method='median')
        self.master_flat.write(self.path_2 + 'master_flat.fits', overwrite=True)
        logging.debug('Master flat created')

    def apply_bias_dark_flat_correction(self):
        # Create a list of the science images
        science_list = glob.glob(self.path_4 + '*.fits')
        science_list.sort()

        for f in science_list:
            ccd = CCDData.read(f, unit=u.adu)
            ccd = subtract_bias(ccd, self.master_bias)
            ccd = subtract_dark(ccd, self.master_dark, exposure_time='EXPOSURE', exposure_unit=u.second)
            ccd = flat_correct(ccd, self.master_flat)
            fits.writeto(self.path_4 + 'd_' + os.path.basename(f), ccd.data, ccd.header, overwrite=True)

    logging.debug('Bias, dark, and flat correction applied')

    def plot_master_image(self, master_image):
        plt.imshow(master_image, cmap='gray', origin='lower', vmin=np.min(master_image), vmax=np.max(master_image))
        plt.colorbar(pad=0.01, fraction=0.047, shrink=1, aspect=20, extend='both', extendrect=True, extendfrac='auto')
        plt.show()

    logging.debug('Master image plotted')


# Paths
path_1 = '/Users/apergis/Downloads/cmos_data//'  # Bias
path_2 = '/Users/apergis/Downloads/cmos_data//'  # Dark
path_3 = '/Users/apergis/Downloads/cmos_data//'  # Flat
path_4 = '/Users/apergis/Downloads/cmos_data//'  # Science

# Dark exposure time for creating master dark
dark_exposure = 10

# Custom keyword for exposure time
exptime_keyword = 'EXPTIME'

# Perform bias, dark, and flat correction
correction = BiasDarkFlatCorrection(path_1, path_2, path_3, path_4)
correction.create_master_bias()
correction.create_master_dark(dark_exposure)
correction.create_master_flat()
correction.apply_bias_dark_flat_correction()

# Plot the master bias image
correction.plot_master_image(correction.master_bias)

# Plot the master dark image
correction.plot_master_image(correction.master_dark)

# Plot the master flat image
correction.plot_master_image(correction.master_flat)
