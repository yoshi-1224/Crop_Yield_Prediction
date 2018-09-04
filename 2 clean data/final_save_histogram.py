import numpy as np
import scipy.io as io
import math


# from clean_data_county import *

class Fetch_data():
    def __init__(self):
        # data dir
        # self.dir = '/atlas/u/jiaxuan/data/MODIS_data/MODIS_data'
        self.dir = "/atlas/u/jiaxuan/data/google_drive/img_output/"
        # self.dir = "/atlas/u/jiaxuan/data/google_drive/img_full_output/"
        # self.dir = 'C:\\0machine_learning\\MODIS_data\\'
        # self.dir = 'F:/0SummerIntern/4_data_input/6_Data_county_processed/'
        # self.dir = 'C:/360Downloads/6_Data_county_processed/'
        # self.dir = '/media/sf_360Downloads/6_Data_county_processed/'

        # output dir
        # self.dir_output = 'C:/360Downloads/6_Data_county_processed_scaled/'
        # self.dir_output = '/atlas/u/jiaxuan/data/MODIS_data_county_processed_scaled/'

        # load yield data
        self.data_yield = np.genfromtxt('yield_final_highquality.csv', delimiter=',')  # label
        self.locations = np.genfromtxt('locations_final.csv', delimiter=',')
        # load soil and weather data
        # self.soil = np.genfromtxt(self.dir+'soil_output.csv', delimiter=',')
        # self.weather = np.genfromtxt(self.dir+'daymet_mean.csv', delimiter=',')

        # generate index for all data
        length = self.data_yield.shape[0]  # number of rows
        self.index_all = np.arange(length)
        # load random data, no need to shuffle
        # np.random.shuffle(self.index_all)

        # # divide all data into 3 groups
        # self.index_train = self.index_all[0:int(length * 0.8)]
        # self.index_val = self.index_all[int(length * 0.8):int(length * 0.9)]
        # self.index_test = self.index_all[int(length * 0.9):length]

        self.year = 2012
        # divide data by year
        self.index_train = []
        self.index_validate = []
        for i in range(length):
            if self.data_yield[i, 0] == self.year:  # the first column is year. 2012 data is used for validation
                self.index_validate.append(i)
            else:
                self.index_train.append(i)
        self.index_train = np.array(self.index_train)
        self.index_validate = np.array(self.index_validate)

        # load mean image
        # try:
        #     self.image_mean = np.load('image_mean_county.npy')
        #     print 'image mean loaded', self.image_mean.shape
        # except:
        #     print 'no mean image found'
        #     self.image_mean = self.calc_mean()
        #     # print self.image_mean

    def get_image_data(self, i):
        year = str(int(self.data_yield[i, 0]))
        loc1 = str(int(self.data_yield[i, 1]))
        loc2 = str(int(self.data_yield[i, 2]))

        key = np.array([int(loc1), int(loc2)])
        index = np.where(np.all(self.locations[:, 0:2].astype('int') == key, axis=1))
        longitude = self.locations[index, 2]
        latitude = self.locations[index, 3]

        filename = year + '_' + loc1 + '_' + loc2 + '.npy'
        image_temp = np.load(self.dir + filename)
        image_temp = self.filter_timespan(image_temp, 49, 305, 9)

        return year, loc1, loc2, longitude, latitude, image_temp

    def calc_mean(self):
        # traverse training data
        image_mean = np.zeros(shape=[32, 32, 9], dtype=np.float32)  # 32 = numBins, band = 9

        print 'traversing training data to calc image_mean:'
        n = 0
        for i in self.index_train:
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            filename = year + '_' + loc1 + '_' + loc2 + '.mat'  # MatLab file.
            content = io.loadmat(self.dir + filename)
            image_temp = content['image_divide']
            image_temp = self.filter_size(image_temp, 48)
            image_temp = self.filter_timespan(image_temp, 49, 305, 7)
            image_temp = self.filter_abnormal(image_temp, 0, 5000)
            image_temp = self.filter_crop_mask(image_temp, year, loc1, loc2)

            ratio = float(n) / float(n + 1)
            image_mean *= ratio
            ratio = float(n + 1)
            image_temp /= ratio
            image_mean += image_temp
            n += 1

            # print ratio
            # print np.mean(image_mean)
            if n % 1000 == 0:
                print n
        # print image_mean
        image_mean = image_mean.astype(dtype='float32', copy=False)
        np.save('image_mean_48by48.npy', image_mean)  # .npy is not a typo, but numpy file.
        return image_mean

    def filter_size(self, image_temp, size):
        return image_temp[0:size, 0:size, :].astype(np.float32)

    def filter_abnormal(self, image_temp, min, max):
        image_temp[image_temp < min] = min
        image_temp[image_temp > max] = max
        return image_temp

    def filter_timespan(self, image_temp, start_day, end_day, bands_per_image):
        interval = 8  # days, given in the paper.
        start_index = int(math.floor(start_day / interval)) * bands_per_image
        end_index = int(math.floor(end_day / interval)) * bands_per_image
        if end_index > image_temp.shape[2]:  # exceeds the total #bands in the image
            image_temp = np.concatenate((image_temp,
                                         np.zeros(shape=(image_temp.shape[0], image_temp.shape[1],
                                                         end_index - image_temp.shape[2]))), axis=2)
        return image_temp[:, :, start_index:end_index]

    def calc_histogram(self, image_temp, bin_seq, bins, times, bands):
        hist = np.zeros(shape=[bins, times, bands])
        for i in range(image_temp.shape[2]):  # bands
            frequency, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)  # density False means frequency.
            # if density.sum()==0:
            #     continue
            hist[:, i / bands, i % bands] = frequency / float(frequency.sum())
        print image_temp.shape[2]
        return hist

    def calc_histogram_flat(self, image_temp, bin_seq, bins, times, bands):
        hist = np.zeros([bins, times, bands])
        for i in range(image_temp.shape[1]):
            density, _ = np.histogram(image_temp[:, i], bin_seq, density=False)
            hist[:, i / bands, i % bands] = density / float(density.sum())
        return hist

    # def calc_histogram(self,image_temp,bin_seq_img,bin_seq_temp,bins,times,bands):
    #     hist=np.zeros([bins,times,bands])
    #     for i in range(image_temp.shape[2]):
    #         if i%bands <=6:
    #             density, _ = np.histogram(image_temp[:, :, i], bin_seq_img, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #         else:
    #             density, _ = np.histogram(image_temp[:, :, i], bin_seq_temp, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #     return hist

    # def calc_histogram_flat(self,image_temp,bin_seq_img,bin_seq_temp,bins,times,bands):
    #     hist=np.zeros([bins,times,bands])
    #     for i in range(image_temp.shape[1]):
    #         if i%bands <=6:
    #             density, _ = np.histogram(image_temp[:, i], bin_seq_img, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #         else:
    #             density, _ = np.histogram(image_temp[:, i], bin_seq_temp, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #     return hist

    def quality_dector(self, image_temp):
        filter_0 = image_temp > 0
        filter_5000 = image_temp < 5000
        filter = filter_0 * filter_5000
        return float(np.count_nonzero(filter)) / image_temp.size

    def next_batch_hist(self, batch_size, type='train'):
        if type == 'train':
            index = self.index_train
        elif type == 'validate':
            index = self.index_validate
        elif type == 'test':
            index = self.index_test
        index_batch = np.random.choice(index, batch_size)
        output_image = np.zeros([batch_size, 32, 32, 9])
        output_yield = np.zeros([batch_size])
        for n, i in enumerate(index_batch):
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            # filename = year + '_' + loc1 + '_' + loc2 + '.mat'
            filename = year + '_' + loc1 + '_' + loc2 + '.npz'
            content = np.load(self.dir + filename)
            image_temp = content['arr_0']
            image_temp = self.filter_timespan(image_temp, 49, 305, 9)

            bin_seq = np.linspace(1, 4999, 33)
            image_temp = self.calc_histogram(image_temp, bin_seq, 32, 32, 9)
            image_temp[np.isnan(image_temp)] = 0
            if np.sum(image_temp) < 250:
                print 'broken image', filename
                print np.isnan(image_temp)

            # output_image[n, :] = image_temp - self.image_mean
            output_image[n, :] = image_temp
            output_yield[n] = self.data_yield[i, 3]
            # print image_temp.shape
            # print np.sum(image_temp)
        return (np.float32(output_yield), np.float32(output_image))

    ## save supervised data
    def save_histogram_supervised(self):
        output_image = np.zeros([self.index_all.shape[0], 32, 32, 9])
        output_yield = np.zeros([self.index_all.shape[0]])
        output_year = np.zeros([self.index_all.shape[0]])
        output_locations = np.zeros([self.index_all.shape[0],2])
        output_index = np.zeros([self.index_all.shape[0],2])

        for i in self.index_all:
            year, loc1, loc2, longitude, latitude, image_temp = self.get_image_data(i)
            print image_temp.shape

            bin_seq = np.linspace(start=1, stop=4999, num=33)  # 50 evenly spaced numbers. Note that 33~32 = bin size.
            image_temp = self.calc_histogram(image_temp, bin_seq, 32, 32, 9)
            image_temp[np.isnan(image_temp)] = 0  # shape is (32, 32, 9)
            # if np.sum(image_temp) < 250:
            #     print 'broken image', filename
            #     print np.isnan(image_temp)

            output_image[i, :] = image_temp
            output_yield[i] = self.data_yield[i, 3]
            output_year[i] = int(year)
            output_locations[i, 0] = longitude
            output_locations[i, 1] = latitude
            output_index[i,:] = np.array([int(loc1),int(loc2)])
            # print image_temp.shape
            print i,np.sum(image_temp),year,loc1,loc2
        np.savez(self.dir+'histogram_all_full.npz',
                 output_image=output_image,output_yield=output_yield,
                 output_year=output_year,output_locations=output_locations,output_index=output_index)
        print 'saved histogram done'


    def save_unsupervised_data(self):
        count_max = 20000
        count = 0
        output_image = np.zeros([count_max, 32, 32, 9])
        output_yield = np.zeros([count_max])
        output_year = np.zeros([count_max])
        output_locations = np.zeros([count_max, 2])
        output_index = np.zeros([count_max, 2])
        for i in self.index_all:
            year, loc1, loc2, longitude, latitude, image_temp = self.get_image_data(i)
            # print image_temp.shape,image_temp.mean()

            image_temp = np.reshape(image_temp, (image_temp.shape[0] * image_temp.shape[1], image_temp.shape[2]),
                                    order='C')
            # remove 0 and 5000
            image_temp[image_temp == 5000] = 0
            # image_temp = image_temp[np.all(image_temp, axis=1)]
            image_temp = image_temp[~np.all(image_temp == 0, axis=1)]
            # print image_temp.shape

            crop_pixel_count = 200
            j = 0
            while j < (image_temp.shape[0] / crop_pixel_count):
                image_temp_part = image_temp[j * crop_pixel_count:(j + 1) * crop_pixel_count, :]
                j += 1
                bin_seq = np.linspace(1, 4999, 33)
                image_temp_part = self.calc_histogram_flat(image_temp_part, bin_seq, 32, 32, 9)
                image_temp_part[np.isnan(image_temp_part)] = 0
                # if np.sum(image_temp_part) < 288:
                #     print 'broken image', filename, np.sum(image_temp_part)
                #     continue

                epoch = count / count_max
                # saver
                if count % count_max == 0 and count != 0:
                    # save
                    np.savez(self.dir + 'histogram_semi_rand_200_20000' + str(epoch) + '.npz',
                             output_image=output_image, output_yield=output_yield,
                             output_year=output_year, output_locations=output_locations, output_index=output_index)
                    print 'save', self.dir + 'histogram_semi_rand_200_20000' + str(epoch) + '.npz'
                    # clear
                    output_image = np.zeros([count_max, 32, 32, 9])
                    output_yield = np.zeros([count_max])
                    output_year = np.zeros([count_max])
                    output_locations = np.zeros([count_max, 2])
                    output_index = np.zeros([count_max, 2])

                output_image[count - epoch * count_max, :] = image_temp_part
                output_yield[count - epoch * count_max] = self.data_yield[i, 3]
                output_year[count - epoch * count_max] = int(year)
                output_locations[count - epoch * count_max, 0] = longitude
                output_locations[count - epoch * count_max, 1] = latitude
                output_index[count - epoch * count_max, :] = np.array([int(loc1), int(loc2)])
                print epoch, i, j, count, np.sum(image_temp_part), year, loc1, loc2
                count += 1
        print 'save done'

    def save_data_mean(self):
        """
        saves np arrays that contain the mean pixel values for all the images in each band. The result is ONLY USED IN ANALYSIS, not to train the model
        """
        output_image = np.zeros(shape=[self.index_all.shape[0], 32 * 9])
        # shape[0] == length == numRows. 32 = #bins, in Experiments section of the report.
        output_yield = np.zeros(shape=[self.index_all.shape[0]])
        output_year = np.zeros(shape=[self.index_all.shape[0]])
        output_locations = np.zeros(shape=[self.index_all.shape[0], 2])  # 2 location values
        output_index = np.zeros(shape=[self.index_all.shape[0], 2])

        for i in self.index_all:  # contains numbers from 0 to length of data
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            # locations.csv format: loc1, loc2, longitude, latitude

            key = np.array([int(loc1), int(loc2)])
            index = np.where(np.all(self.locations[:, 0:2].astype('int') == key,
                                    axis=1))  # np.all returns array of boolean telling if ALL (as opposed to ANY) elements along the given axis evaluates to True for the given condition. np.where returns the INDEX of the entries. Note that for locations, we only have 2 axis because it is just a table.
            longitude = self.locations[index, 2]
            latitude = self.locations[index, 3]

            filename = year + '_' + loc1 + '_' + loc2 + '.npy'
            image_temp = np.load(
                self.dir + filename)  # the files were already saved as .npy before this. We know the year of the image by the filename
            image_temp = self.filter_timespan(image_temp, start_day=49, end_day=305, bands_per_image=9)

            image_temp = np.sum(image_temp, axis=(0, 1)) / np.count_nonzero(image_temp) * image_temp.shape[
                2]  # this multiplication is here because np.count_nonzero counts non-zeros in all dimensions, even though we only sum across height & width dimensions.

            print 'The shape of image_temp after np.sum is %s' % str(image_temp.shape)
            # image_temp should contain an array of len(bands) * length of time
            image_temp[np.isnan(image_temp)] = 0

            output_image[i, :] = image_temp  # TODO: vector of length = 32 * 9 or what?
            output_yield[i] = self.data_yield[i, 3]
            output_year[i] = int(year)
            output_locations[i, 0] = longitude
            output_locations[i, 1] = latitude
            output_index[i, :] = np.array([int(loc1), int(loc2)])
            # print image_temp.shape
            print i, np.sum(image_temp), year, loc1, loc2
        np.savez(self.dir + 'histogram_all_mean.npz',  ## TODO this file is only used in result analysis, not for training.
                 output_image=output_image, output_yield=output_yield,
                 output_year=output_year, output_locations=output_locations,
                 output_index=output_index)  # zipped group of arrays. These arrays can be accessed like a dictionary value with its name as the key.
        print 'save done'


if __name__ == '__main__':
    data = Fetch_data()  # instance of class Fetch_data
    # calculate mean
    # data.calc_mean()

    # output_yield, output_image=data.next_batch(8,'train')
    # print output_yield.shape,output_yield
    # print output_image.shape,output_image

    # preprocess_save_data()
    # label,img=data.next_batch_hist(data.index_test.size,'test')
    # i=0
    # print img[i].shape, img[i].dtype, img[i][:,0,0].sum()

    # check_data_integrity()
    # data.calc_mean()
    # i,a=data.next_batch_hist(32,'train')
    data.save_histogram_supervised()
    # data.save_data()
    # data.save_data_mean()
    # data.save_data('validate')
    # data.save_data('test')
