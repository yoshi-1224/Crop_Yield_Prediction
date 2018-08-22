import ee
import pandas as pd
from pull_MODIS import appendBand_6

ee.Initialize()

# locations = pd.read_csv('locations_remedy.csv')
locations = pd.read_csv('world_locations.csv', header=None)

# county_region = ee.FeatureCollection('ft:18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp')
world_region = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')

imgcoll = ee.ImageCollection('MODIS/MOD09A1') \
    .filterDate('2001-12-31', '2015-12-31')
img = imgcoll.iterate(appendBand_6)
img = ee.Image(img)

for country, index in locations.values:
    scale = 500
    crs = 'EPSG:4326'

    # filter for a county
    region = world_region.filterMetadata('Country', 'equals', country)
    if region is None: # not region == None, although semantically they're the same
        print country, index, 'not found'
        continue
    region = region.first()
    # region = region.geometry().coordinates().getInfo()[0]

    img_temp = img.clip(region)
    hist = ee.Feature(None, {
        'mean': img_temp.reduceRegion(ee.Reducer.fixedHistogram(1, 4999, 32), region, scale, crs, None, False, 1e12,
                                      16)})

    hist_info = hist.getInfo()['features']
    print hist_info
