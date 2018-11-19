from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

file_name = "1_32.tif"

tiff_file = gdal.Open(file_name)
band1 = tiff_file.GetRasterBand(1000)
img = band1.ReadAsArray(0, 0, tiff_file.RasterXSize, tiff_file.RasterYSize)

max_pixel_value = 2000
scale = max_pixel_value / 255

img /= scale
# img = np.dstack((img / scale, img / scale, img / scale))  # this only makes sense for 3bands

f = plt.figure()
plt.imshow(img)
plt.show()
