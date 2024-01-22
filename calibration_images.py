import glob
import os
from astropy.io import fits
import numpy as np
import re


def bias(base_path, out_path):
    if os.path.exists(out_path + 'master_bias.fits'):
        print('Master bias already exists')
        pass

    # Find and read the bias for hdr mode
    files = [f for f in glob.glob(base_path + 'bias*.fits') if 'HDR' in fits.getheader(f)['READMODE']]

    # Limit the number of files to the first 21
    files = files[:21]

    cube = np.zeros((2048, 2048, len(files)))
    for i, f in enumerate(files):
        cube[:, :, i] = fits.getdata(f)
    master_bias = np.median(cube, axis=2)
    fits.PrimaryHDU(master_bias).writeto('master-bias.fits', overwrite=True)
    return master_bias


def dark(base_path, out_path, master_bias):
    if os.path.exists(out_path + 'master_dark.fits'):
        print('Master dark already exists')
        pass

    # Find and read the darks for hdr mode
    files = [f for f in glob.glob(base_path + 'dark*.fits') if 'HDR' in fits.getheader(f)['READMODE']]

    # Limit the number of files to the first 21
    files = files[:21]

    cube = np.zeros((2048, 2048, len(files)))
    for i, f in enumerate(files):
        cube[:, :, i] = fits.getdata(f)
    master_dark = np.median(cube, axis=2) - master_bias
    fits.PrimaryHDU(master_dark).writeto('master-dark.fits', overwrite=True)
    return master_dark


def flat(base_path, out_path, master_bias, master_dark, dark_exposure=10, prefix):
    if os.path.exists(out_path + 'master_flat.fits'):
        print('Master flat already exists')
        pass
    # Mask out the non-imaging areas of the sensor
    h = fits.getheader(glob.glob(f'/data/20240109/evening-flat-*-*.fits')[0])
    r = re.search(r'^\[(\d+):(\d+),(\d+):(\d+)\]$', h['IMAG-RGN']).groups()
    x1 = int(r[0]) - 1
    x2 = int(r[1])
    y1 = int(r[2]) - 1
    y2 = int(r[3])

    mask = np.ones(master_bias.shape, dtype=bool)
    mask[y1:y2, x1:x2] = 0

    for filt in ['B', 'V', 'R', 'I', 'NONE']:
        print(f'Creating {filt} master flat')
        files = glob.glob(f'/data/20240109/evening-flat-{filt}-*.fits')
        if files:
            cube = np.zeros((*master_bias.shape, len(files)))
            for i, f in enumerate(files):
                data, header = fits.getdata(f, header=True)
                cube[:, :, i] = data - master_bias - master_dark * header['EXPTIME'] / dark_exposure
                cube[:, :, i] = cube[:, :, i] / np.average(cube[:, :, i])
                cube[mask, i] = 1

            master_flat = np.median(cube, axis=2)
            fits.PrimaryHDU(master_flat).writeto(f'master-flat-{filt}.fits', overwrite=True)
            del cube
            del master_flat


if __name__ == '__main__':
    path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/20231212/'
    out_path = '/Users/u5500483/Documents/GitHub/ngcmos/'
    master_bias = bias(path, out_path)
    dark(path, out_path, master_bias)