from astropy.io import fits

# Open the FITS file
with fits.open('your_file.fits') as hdul:
    # Print the information about the file
    hdul.info()

    # Access specific HDU (Header Data Unit)
    # For example, if the first HDU is a table:
    table_hdu = hdul[1]

    # Access data from the HDU
    data = table_hdu.data
    header = table_hdu.header

    # Work with the data and header as needed
    print(data)
    print(header)