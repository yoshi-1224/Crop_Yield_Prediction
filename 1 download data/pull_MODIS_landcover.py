import ee
import time
import pandas as pd
from pull_MODIS import export_oneimage, appendBand_0

"""
MODIS landcover provides classification of land types: cropland, grassland, sea, etc. This script is to pull that data to use it as a mask.
"""

ee.Initialize()
locations = pd.read_csv('locations.csv')

imgcoll = ee.ImageCollection('MODIS/051/MCD12Q1') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23))
img=imgcoll.iterate(appendBand_0)

for loc1, loc2, lat, lon in locations.values:
    file_name = '{}_{}'.format(int(loc1), int(loc2))

    offset = 0.11
    scale  = 500
    crs='EPSG:4326'

    region = str([
        [lat - offset, lon + offset],
        [lat + offset, lon + offset],
        [lat + offset, lon - offset],
        [lat - offset, lon - offset]])

    while True:
        try:
            export_oneimage(img,'Data_mask', file_name, region, scale, crs)
        except:
            print 'retry'
            time.sleep(10)
            continue
        break