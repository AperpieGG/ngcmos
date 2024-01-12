#!/Users/u5500483/anaconda3/bin/python
from datetime import datetime

from donuts import Donuts
import glob
from matplotlib import pyplot as plt


def plot_images():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


plot_images()

path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/TOI-00451/'

reference_image_name = [f for f in glob.glob(path + '*_r.fits')][0]
science_image_names = [f for f in glob.glob(path + '*_r.fits')[150:]]
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

x_shifts = []
y_shifts = []

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

    # Append shift values to the lists
    x_shifts.append(x.value)
    y_shifts.append(y.value)

fig = plt.figure(figsize=(8, 8))
plt.scatter(x_shifts, y_shifts, label='Shifts', marker='o')
plt.xlabel('X Shift (pixels)')
plt.ylabel('Y Shift (pixels)')
plt.title('Shifts with respect to the ref image')
plt.axhline(0, color='black', linestyle='-', linewidth=1)  # Add horizontal line at y=0
plt.axvline(0, color='black', linestyle='-', linewidth=1)  # Add vertical line at x=0
plt.legend()

# Set the axes limits to center (0, 0)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

save_path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/TOI-00451/'
fig.savefig(save_path + "shifts_{}.pdf".format(datetime.now().strftime("%Y%m%d-%H%M%S")), bbox_inches='tight')
