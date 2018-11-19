import ee
import time
import pandas as pd
from pull_MODIS import export_oneimage, appendBand_0

ee.Initialize()
locations = pd.read_csv('locations_final_1.csv')

county_region = ee.FeatureCollection('ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM')

imgcoll = ee.ImageCollection('MODIS/051/MCD12Q1') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50, -64, 23)) \
    .filterDate('2002-12-31', '2016-8-4')
img = imgcoll.iterate(appendBand_0)
img = ee.Image(img)

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
    scale = 500
    crs = 'EPSG:4326'

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
            export_oneimage(img.clip(region), 'data_mask', fname, scale, None, crs) # TODO: None. Seems like 'clip.py' has no region

        except:
            print 'retry'
            time.sleep(10)
            continue
        break