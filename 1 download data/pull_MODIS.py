import ee
import time
import pandas as pd

def export_oneimage(img, folder, name, region, scale, crs):
    # task = ee.batch.Export.image(image=img, description=name, config={
    #     'driveFolder':folder,
    #     'driveFileNamePrefix':name,
    #     'region': region,
    #     'scale':scale,
    #     'crs':crs
    # }) # this is somehow raised because we are trying to call __new__ but __init__ gets in the way
    # and it is deprecated: use Export.image.toDrive or Export.image.toCloudStorage or toAsset

    task = ee.batch.Export.image.toDrive(image=img, description=name, folder=folder, region=region, scale=scale, crs=crs)
    task.start()
    while task.status()['state'] == 'RUNNING':  # or READY or COMPLETE
        print 'Running...'
        # Perhaps task.cancel() at some point.
        time.sleep(10)
    print 'Done.', task.status()


def appendBand_6(current, previous):  # reduce function, not map. So current means it is the accumulated collection
    """
    Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    Author: Jamie Vleeshouwer
    """
    # Rename the band
    previous = ee.Image(previous)
    current = current.select([0, 1, 2, 3, 4, 5, 6])  # selects the bands from image. get method is for getting properties from a feature.
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(
        ee.Image(current)))  # if previous is None then return current, otherwise add current to previous. (Note: only return current item on first element/iteration)
    return accum

def appendBand_0(current, previous):
    previous = ee.Image(previous)
    current = current.select([0])  # selects the bands from image. get method is for getting properties from a feature.
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(
        ee.Image(current)))  # if previous is None then return current, otherwise add current to previous. (Note: only return current item on first element/iteration)
    return accum

def appendBand_temperature(current, previous):
    previous = ee.Image(previous)
    current = current.select([0, 4])  # selects the bands from image. get method is for getting properties from a feature.
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(
        ee.Image(current)))  # if previous is None then return current, otherwise add current to previous. (Note: only return current item on first element/iteration)
    return accum

ee.Initialize()  # Google earth-engine API. Authentication etc.

print 'Initialisation of ee is complete. Now reading csv file'
locations = pd.read_csv('locations.csv')

imgcoll = ee.ImageCollection('MODIS/MOD09A1').filterBounds(ee.Geometry.Rectangle(-106.5, 50, -64, 23))
img = imgcoll.iterate(appendBand_6)

for loc1, loc2, lat, lon in locations.values:
    file_name = '{}_{}'.format(int(loc1), int(loc2))

    offset = 0.11
    scale = 500
    crs = 'EPSG:4326'  # crs = "Coordinate Reference System"

    region = str([
        [lat - offset, lon + offset],
        [lat + offset, lon + offset],
        [lat + offset, lon - offset],
        [lat - offset, lon - offset]])

    print 'region is %s' % region

    while True:
        try:
            export_oneimage(img, 'Data_folder', file_name, region, scale, crs)
        except:
            print 'retry'
            time.sleep(10)
            continue
        break

# ee.Initialize()
# for attr in dir(ee.ImageCollection('MODIS/MOD09A1')):
#     if "image" in attr.lower():
#         print attr
