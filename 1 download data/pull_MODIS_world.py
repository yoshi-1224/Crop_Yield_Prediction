import ee # earth engine
import time
import pandas as pd
from pull_MODIS import export_oneimage, appendBand_6

ee.Initialize()

# locations = pd.read_csv('locations_remedy.csv')
locations = pd.read_csv('world_locations.csv',header=None)

# county_region = ee.FeatureCollection('ft:18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp')
world_region = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')

imgcoll = ee.ImageCollection('MODIS/MOD09A1') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23))\
    .filterDate('2001-12-31','2015-12-31')

img=imgcoll.iterate(appendBand_6)
img=ee.Image(img)

img_0=ee.Image(ee.Number(0))
img_5000=ee.Image(ee.Number(5000))

img=img.min(img_5000)
img=img.max(img_0)

# img=ee.Image(ee.Number(100))
# img=ee.ImageCollection('LC8_L1T').mosaic()

for country,index in locations.values:
    fname = 'index'+'{}'.format(int(index))

    # offset = 0.11
    scale  = 500
    crs='EPSG:4326'

    # filter for a county
    region = world_region.filterMetadata('Country', 'equals', country)
    if region==None:
        print country,index,'not found'
        continue
    region = region.first()
    region = region.geometry().coordinates().getInfo()[0]

    # region = str([
    #     [lat - offset, lon + offset],
    #     [lat + offset, lon + offset],
    #     [lat + offset, lon - offset],
    #     [lat - offset, lon - offset]])
    while True:
        try:
            export_oneimage(img, 'Data_world', fname, region, scale, crs)
        except:
            print 'retry'
            time.sleep(10)
            continue
        # break