import ee
import time
import pandas as pd
from pull_MODIS import appendBand_temperature, export_oneimage

ee.Initialize()

locations = pd.read_csv('locations_final.csv')

county_region = ee.FeatureCollection('ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM')

imgcoll = ee.ImageCollection('MODIS/MYD11A2') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23))\
    .filterDate('2002-12-31','2016-8-4')
img=imgcoll.iterate(appendBand_temperature)
img=ee.Image(img)

# img_0=ee.Image(ee.Number(0))
# img_5000=ee.Image(ee.Number(5000))
#
# img=img.min(img_5000)
# img=img.max(img_0)

# img=ee.Image(ee.Number(100))
# img=ee.ImageCollection('LC8_L1T').mosaic()

for loc1, loc2, lat, lon in locations.values:
    fname = '{}_{}'.format(int(loc1), int(loc2))

    # offset = 0.11
    scale  = 500
    crs='EPSG:4326'

    # filter for a county
    region = county_region.filterMetadata('StateFips', 'equals', int(loc1))
    region = ee.FeatureCollection(region).filterMetadata('CntyFips', 'equals', int(loc2))
    region = ee.Feature(region.first())

    # region = str([
    #     [lat - offset, lon + offset],
    #     [lat + offset, lon + offset],
    #     [lat + offset, lon - offset],
    #     [lat - offset, lon - offset]])
    while True:
        try:
            # export_oneimage(img, 'Data_test', fname, region, scale, crs)
            export_oneimage(img.clip(region), 'data_temperature', fname, scale, crs)
        except:
            print 'retry'
            time.sleep(10)
            continue
        break