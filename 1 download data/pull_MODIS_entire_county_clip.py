import ee
import time
import pandas as pd
from pull_MODIS import export_oneimage, appendBand_6

ee.Initialize()
locations = pd.read_csv('locations_final.csv', header=None)

county_region = ee.FeatureCollection('ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM')

imgcoll = ee.ImageCollection('MODIS/MOD09A1') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50, -64, 23)) \
    .filterDate('2002-12-31', '2016-8-4')
img = imgcoll.iterate(appendBand_6)
img = ee.Image(img)

img_0 = ee.Image(ee.Number(-100))
img_16000 = ee.Image(ee.Number(16000))

img = img.min(img_16000)
img = img.max(img_0)

# img=ee.Image(ee.Number(100))
# img=ee.ImageCollection('LC8_L1T').mosaic()

for loc1, loc2, lat, lon in locations.values:
    file_name = '{}_{}'.format(int(loc1), int(loc2))

    # offset = 0.11
    scale = 500
    crs = 'EPSG:4326'

    # filter for a county
    region = county_region.filterMetadata('StateFips', 'equals', int(loc1))
    region = ee.FeatureCollection(region).filterMetadata('CntyFips', 'equals', int(loc2))
    region = ee.Feature(region.first())
    # region = region.geometry().coordinates().getInfo()[0]

    # region = str([
    #     [lat - offset, lon + offset],
    #     [lat + offset, lon + offset],
    #     [lat + offset, lon - offset],
    #     [lat - offset, lon - offset]])
    while True:
        try:
            export_oneimage(img.clip(region), 'test', file_name, scale, None, crs) # TODO: None for region
        except:
            print 'retry'
            time.sleep(10)
            continue
        break
    # while True:
    #     try:
    #         export_oneimage(img,'Data_test',fname,region,scale,crs)
    #     except:
    #         print 'retry'
    #         time.sleep(10)
    #         continue
    #     break
