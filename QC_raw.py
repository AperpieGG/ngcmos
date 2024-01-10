from astroquery.astrometry_net import AstrometryNet, core
from astropy.io import fits
import glob


def download_wcs_solution(image_path, api_key):
    # Set Astrometry.net API key
    AstrometryNet.key = api_key

    # Loop over all FITS files in the directory
    for image_path in images:
        # Open the FITS file and extract the header

        with fits.open(image_path, mode='update') as hdulist:
            print('Updating WCS for {}'.format(image_path))
            wcs_header = hdulist[0].header
            # Use Astrometry.net to solve WCS
            ast = AstrometryNet()
            # Specify the format when reading the FITS file
            result = ast.solve_from_image(image_path)
            print('result and type: {} {}'.format(result, type(result)))

            # Update the WCS header with the new solution
            wcs_header.update(result)

            # Save the updated FITS file
            hdulist.flush()


if __name__ == "__main__":
    # Replace these with your actual directory path and Astrometry.net API key
    directory_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/TOI-00451/'
    images = [f for f in glob.glob(directory_path + '*.fits')]
    astrometry_net_api_key = "xmurixquvwrirexd"

    download_wcs_solution(directory_path, astrometry_net_api_key)
