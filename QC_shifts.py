from donuts import Donuts
import glob


path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/TOI-00451/'

reference_image_name = [f for f in glob.glob(path + '*_r.fits')][0]
science_image_names = [f for f in glob.glob(path + '*_r.fits')[1:]]
# Construct a donuts object
d = Donuts(
    refimage=reference_image_name,
    image_ext=0,
    overscan_width=20,
    prescan_width=20,
    border=64,
    normalise=True,
    exposure='EXPTIME',
    subtract_bkg=True,
    ntiles=32)
# for each image, compute the x/y translation required
# to align the images onto the reference image
for image in science_image_names:
    shift_result = d.measure_shift(image)
    x = shift_result.x
    y = shift_result.y
    # Also check out shift_result.sky_background
    print(x, y)

    if x.value > 0.5 or y.value > 0.5 or x.value < -0.5 or y.value < -0.5:
        print('WARNING: Image {} is not aligned'.format(image))
    else:
        pass

