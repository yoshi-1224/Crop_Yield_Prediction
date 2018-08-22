import numpy as np
import scipy.io as io  # for loading matlab files
import os
import \
    gdal  # Python automatically calls GDALAllRegister() when the gdal module is imported, so can already call gdal.Open
from scipy.ndimage import zoom
from joblib import Parallel, delayed  # for multi-threading
from utils import constants


################
# Data range
# MODIS: 2003-2016, 14 years
# MODIS_landcover: 2003-2013, 12 years
# MODIS_temperature: 2003_2015, 13 years

# Intersection: 2003-2013, 11 years: TODO but can't we do these kind of preprocessing with EarthEngine itself, rather than doing it manually?
################


def check_data_integrity_del(mode='read'):
    data = np.genfromtxt(fname='yield_final_highquality.csv', delimiter=',')
    # check if they have related files
    dir = "/atlas/u/jiaxuan/data/google_drive/img_zoom_output/"
    list_del = []
    for i in range(data.shape[0]):  # the first dimension of data == number of rows
        year = data[i, 0]  # this corresponds to the csv format
        loc1 = data[i, 1]
        loc2 = data[i, 2]
        filename = str(int(year)) + '_' + str(int(loc1)) + '_' + str(int(loc2)) + '.npy'
        if not os.path.isfile(dir + filename):
            if mode == 'read':
                print filename
            elif mode == 'delete':
                print 'del'
                list_del.append(i)

    if mode == 'delete':
        list_del = np.array(list_del)
        data_clean = np.delete(data, list_del, axis=0)
        np.savetxt(fname="yield_final_highquality.csv", X=data_clean, delimiter=",")


def divide_image(img, first, step, num):
    image_list = []
    for i in xrange(0, num - 1):
        image_list.append(img[:, :, first:first + step])
        first += step
    image_list.append(img[:, :, first:])
    return image_list


def extend_mask(img, num):
    for i in range(0, num):
        img = np.concatenate((img, img[:, :, -2:-1]), axis=2)
    return img


# very dirty... but should work
def merge_image(MODIS_img_list, MODIS_temperature_img_list):
    MODIS_list = []
    for i in range(0, len(MODIS_img_list)):
        img_shape = MODIS_img_list[i].shape  # tuple storing the length of each dimension
        img_temperature_shape = MODIS_temperature_img_list[i].shape
        img_shape_new = (img_shape[0], img_shape[1], img_shape[2] + img_temperature_shape[2])
        # img_shape[0] and [1] should be width-height, same as the dimension of temperature
        # shape[2] should be the band. 7 + 2 = 9 bands [0:6, 0, 4]

        merge = np.empty(img_shape_new)
        for j in range(0, img_shape[2] / 7):  # 7 is the number of bands? then j is the size of each band?
            img = MODIS_img_list[i][:, :, (j * 7):(j * 7 + 7)]
            temperature = MODIS_temperature_img_list[i][:, :, (j * 2):(j * 2 + 2)] # 2 is also the number of bands right?
            merge[:, :, (j * 9):(j * 9 + 9)] = np.concatenate((img, temperature), axis=2)
        MODIS_list.append(merge)
    return MODIS_list


def mask_image(MODIS_list, MODIS_mask_img_list):
    MODIS_list_masked = []
    for i in range(0, len(MODIS_list)):
        mask = np.tile(MODIS_mask_img_list[i], reps=(1, 1, MODIS_list[i].shape[2]))  # reps = #repeats
        masked_img = MODIS_list[i] * mask  # element-wise multiplication. Width-height should be the same, so just repeat it across different bands
        MODIS_list_masked.append(masked_img)
    return MODIS_list_masked


def quality_dector(image_temp):
    filter_0 = image_temp > 0
    filter_5000 = image_temp < 5000
    filter = filter_0 * filter_5000
    return float(np.count_nonzero(filter)) / image_temp.size


def preprocess_save_data_parallel(file_name):
    # MODIS_processed_dir="C:/360Downloads/6_Data_county_processed_scaled/"
    # MODIS_dir="/atlas/u/jiaxuan/data/MODIS_data_county/3_Data_county"
    # MODIS_temperature_dir="/atlas/u/jiaxuan/data/MODIS_data_county_temperature"
    # MODIS_mask_dir="/atlas/u/jiaxuan/data/MODIS_data_county_mask"
    # MODIS_processed_dir="/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/"

    if file_name.endswith(".tif"):
        MODIS_path = os.path.join(MODIS_dir, file_name)
        # check file size to see if it's broken
        # if os.path.getsize(MODIS_path) < 10000000:
        #     print 'file broken, continue'
        #     continue
        MODIS_temperature_path = os.path.join(constants.MODIS_TEMP_DIR, file_name)
        MODIS_mask_path = os.path.join(constants.MODIS_MASK_DIR, file_name)

        # get geo location
        raw = file_name.replace('_', ' ').replace('.', ' ').split()
        loc1 = int(raw[0])  # not latitude
        loc2 = int(raw[1])  # not longitude
        # read image
        try:
            MODIS_img = np.transpose(np.array(gdal.Open(MODIS_path).ReadAsArray(), dtype='uint16'), axes=(1, 2, 0))
        except ValueError as msg:
            print '%s encountered error' % MODIS_path
            print msg
            return

        # read temperature
        MODIS_temperature_img = np.transpose(np.array(gdal.Open(MODIS_temperature_path).ReadAsArray(), dtype='uint16'),
                                             axes=(1, 2, 0))  # basically the same code as above, except the path

        # # shift
        # MODIS_temperature_img = MODIS_temperature_img - 12000
        # # scale
        # MODIS_temperature_img = MODIS_temperature_img * 1.25
        # # clean
        # MODIS_temperature_img[MODIS_temperature_img < 0] = 0
        # MODIS_temperature_img[MODIS_temperature_img > 5000] = 5000

        # read mask
        MODIS_mask_img = np.transpose(np.array(gdal.Open(MODIS_mask_path).ReadAsArray(), dtype='uint16'),
                                      axes=(1, 2, 0))
        # Non-crop = 0, crop = 1
        MODIS_mask_img[MODIS_mask_img != 12] = 0
        MODIS_mask_img[MODIS_mask_img == 12] = 1

        # Divide image into years # what does the image contain?
        MODIS_img_list = divide_image(MODIS_img, first=0, step=46 * 7, num=14)
        MODIS_temperature_img_list = divide_image(MODIS_temperature_img, first=0, step=46 * 2, num=14)
        MODIS_mask_img = extend_mask(MODIS_mask_img, num=3)
        MODIS_mask_img_list = divide_image(MODIS_mask_img, first=0, step=1, num=14)

        # Merge image and temperature
        MODIS_list = merge_image(MODIS_img_list, MODIS_temperature_img_list)

        # Do the mask job
        MODIS_list_masked = mask_image(MODIS_list, MODIS_mask_img_list)

        # check if the result is in the list
        year_start = 2003
        for i in range(0, 14):
            year = i + year_start
            key = np.array([year, loc1, loc2])
            if np.sum(np.all(data_yield[:, 0:3] == key, axis=1)) > 0:
                # # detect quality
                # quality = quality_dector(MODIS_list_masked[i])
                # if quality < 0.01:
                #     print 'omitted'
                #     print year,loc1,loc2,quality

                # # delete
                # yield_all = np.genfromtxt('yield_final_highquality.csv', delimiter=',')
                # key = np.array([year,loc1,loc2])
                # index = np.where(np.all(yield_all[:,0:3] == key, axis=1))
                # yield_all=np.delete(yield_all, index, axis=0)
                # np.savetxt("yield_final_highquality.csv", yield_all, delimiter=",")

                # continue

                ## 1 save original file
                filename = constants.IMG_OUTPUT_DIR + str(year) + '_' + str(loc1) + '_' + str(loc2) + '.npy'
                np.save(filename, MODIS_list_masked[i])
                print filename, ':written in original folder'

                ## 2 save zoomed file (48*48)
                zoom0 = float(48) / MODIS_list_masked[i].shape[0]
                zoom1 = float(48) / MODIS_list_masked[i].shape[1]
                output_image = zoom(MODIS_list_masked[i], (zoom0, zoom1, 1))

                filename = constants.IMG_ZOOM_OUTPUT_DIR + str(year) + '_' + str(loc1) + '_' + str(loc2) + '.npy'
                np.save(filename, output_image)
                print filename, ':written in zoom folder'


if __name__ == "__main__":
    data_yield = np.genfromtxt('yield_final.csv', delimiter=',', dtype=float)
    MODIS_dir = "/atlas/u/jiaxuan/data/google_drive/data_image_full"
    for _, _, files in os.walk(MODIS_dir):  # root, dir and file respectively
        Parallel(n_jobs=12)(delayed(preprocess_save_data_parallel)(image_file) for image_file in files)

    # # clean yield (low quality)
    # check_data_integrity_del()
    # # check integrity
    # check_data_integrity()

# ctrl+shift+space to get the documentation.
