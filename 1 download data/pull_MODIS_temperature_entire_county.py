import ee
import time
import pandas as pd
from pull_MODIS import export_oneimage, appendBand_temperature

ee.Initialize()
locations = pd.read_csv('locations_major.csv')

county_region = ee.FeatureCollection('ft:18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp')

imgcoll = ee.ImageCollection('MODIS/MYD11A2') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50, -64, 23)) \
    .filterDate('2002-12-31', '2015-12-31')
img = imgcoll.iterate(appendBand_temperature)
img = ee.Image(img)

# img_0=ee.Image(ee.Number(0))
# img_5000=ee.Image(ee.Number(5000))
#
# img=img.min(img_5000)
# img=img.max(img_0)

# img=ee.Image(ee.Number(100))
# img=ee.ImageCollection('LC8_L1T').mosaic()

for loc1, loc2, lat, lon in locations.values:
    file_name = '{}_{}'.format(int(loc1), int(loc2))

    offset = 0.11
    scale = 500
    crs = 'EPSG:4326'

    # filter for a county
    region = county_region.filterMetadata('STATE num', 'equals', loc1)
    region = ee.FeatureCollection(region).filterMetadata('COUNTY num', 'equals', loc2)
    region = region.first()
    region = region.geometry().coordinates().getInfo()[0]

    # region = str([
    #     [lat - offset, lon + offset],
    #     [lat + offset, lon + offset],
    #     [lat + offset, lon - offset],
    #     [lat - offset, lon - offset]])
    while True:
        try:
            export_oneimage(img, 'Data_county_temperature', file_name, region, scale, crs)
        except:
            print 'retry'
            time.sleep(10)
            continue
        break
