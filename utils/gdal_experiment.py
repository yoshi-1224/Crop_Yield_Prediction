from osgeo import gdal  # import gdal is DEPRECATED, although it still works
import numpy as np

file_name = "1_32.tif"

try:
    gtif = gdal.Open(file_name)
    band1 = gtif.GetRasterBand(1)  # gdal is 1-based
    print 'the type of band is %s' % type(band1)
    nBands = gtif.RasterCount  # how many bands, to help you loop
    nRows = gtif.RasterYSize  # how many rows
    nCols = gtif.RasterXSize  # how many columns
    dType = band1.DataType

    print 'the number of bands is %d\nrow = %d\ncol = %d' % (nBands, nRows, nCols)
    print 'the DataType (dtype) of band1 is %s' % dType
    print gtif.GetMetadata()  # notice that there are sooo many bands: we combined many images as bands to form this one big image. Normally, we only have up to 7.

    gtif_array = gtif.ReadAsArray()
    print 'the type of gtif_array is %s' % type(gtif_array)

    gtif_nparray = np.array(gtif_array, dtype='uint16')
    print 'shape of nparray is %s' % str(gtif_nparray.shape)


    MODIS_img = np.transpose(gtif_nparray, axes=(1, 2, 0))
    print 'shape of nparray %s ' % str(MODIS_img.shape)  # L means long type. Not important
except ValueError as msg:
    print "error while reading."
    print msg

finally:
    gtif = None  # close the raster

