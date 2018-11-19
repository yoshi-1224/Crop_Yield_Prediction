from nnet_model import *
from GP_crop_v3 import *
import logging
from utils import constants

# import tensorflow as tf # tensorflow cannot be installed for Python2.7 on Windows

model_to_train = 'lstm'  # or 'conv'


def remove_counties(image_all, yield_all, year_all, locations_all, index_all):
    """
    removes all the counties not used for the model
    """
    counties_keep = (5, 17, 18, 19, 20, 27, 29, 31, 38, 39, 46)  # they filtered all but 11 states
    list_keep = []
    for i in range(image_all.shape[0]):
        if index_all[i][0] in counties_keep:
            list_keep.append(i)

    image_selected = image_all[list_keep, :, :, :]  # TODO why 4D? => 3D histogram, with the freq as another dimension.
    yield_selected = yield_all[list_keep]
    year_selected = year_all[list_keep]
    locations_selected = locations_all[list_keep, :]
    index_selected = index_all[list_keep, :]

    return image_selected, yield_selected, year_selected, locations_selected, index_selected


def load_data(file_path):
    """
    Load npz file and remove all the broken images
    """
    # load data to memory

    # npz is zipped arrays. Think of content variable as a dictionary storing multiple arrays
    content = np.load(file_path)

    image_raw = content['output_image']  # this is an array of 3D histograms
    yield_raw = content['output_yield']
    year_raw = content['output_year']
    locations_raw = content['output_locations']
    index_raw = content['output_index']
    list_delete = []
    for i in range(image_raw.shape[0]):  # number of rows
        if np.sum(image_raw[i, :, :, :]) <= 287:  # image_raw is 4D => list of 3D histograms
            if year_raw[i] < 2016:  # 2016 is used for testing purposes.
                list_delete.append(i)
    image_selected = np.delete(image_raw, list_delete, axis=0)
    yield_selected = np.delete(yield_raw, list_delete, axis=0)
    year_selected = np.delete(year_raw, list_delete, axis=0)
    locations_selected = np.delete(locations_raw, list_delete, axis=0)
    index_selected = np.delete(index_raw, list_delete, axis=0)

    return remove_counties(image_selected, yield_selected, year_selected, locations_selected, index_selected)


def try_restore_model():
    try:
        saver.restore(sess, Config.save_path + str(predict_year) + "CNN_model.ckpt")
        # Restore log results
        npzfile = np.load(Config.save_path + str(predict_year) + 'result.npz')
        summary_train_loss = npzfile['summary_train_loss'].tolist()
        summary_eval_loss = npzfile['summary_eval_loss'].tolist()
        summary_RMSE = npzfile['summary_RMSE'].tolist()
        summary_ME = npzfile['summary_ME'].tolist()
        print("Model restored.")
        return summary_train_loss, summary_eval_loss, summary_RMSE, summary_ME
    except:
        print 'No history model found. Returning None'
        return [], [], [], []


def plot_result():
    # Plot the points using matplotlib
    npzfile = np.load(Config.save_path + str(predict_year) + 'result.npz')
    plot_train_loss = npzfile['summary_train_loss']
    plot_eval_loss = npzfile['summary_eval_loss']
    plot_RMSE = npzfile['summary_RMSE']
    plot_ME = npzfile['summary_ME']

    plt.plot(range(len(plot_train_loss)), plot_train_loss)
    plt.plot(range(len(plot_eval_loss)), plot_eval_loss)
    plt.xlabel('Training steps')
    plt.ylabel('L2 loss')
    plt.title('Loss curve')
    plt.legend(['Train', 'Validate'])
    plt.show()

    plt.plot(range(len(plot_RMSE)), plot_RMSE)
    # plt.plot(range(len(summary_ME)), summary_ME)
    plt.xlabel('Training steps')
    plt.ylabel('Error')
    plt.title('RMSE')
    # plt.legend(['RMSE', 'ME'])
    plt.show()

    # plt.plot(range(len(summary_RMSE)), summary_RMSE)
    plt.plot(range(len(plot_ME)), plot_ME)
    plt.xlabel('Training steps')
    plt.ylabel('Error')
    plt.title('ME')
    # plt.legend(['RMSE', 'ME'])
    plt.show()


if __name__ == "__main__":
    predict_year = 2013
    logging.basicConfig(filename='train_for_hist_alldata' + str(predict_year) + '.log', level=logging.DEBUG)

    filename = constants.HISTOGRAM_32_FILENAME
    # filename = 'histogram_all_soilweather' + '.npz'
    file_path = Config.load_path + filename

    image_all, yield_all, year_all, locations_all, index_all = load_data(file_path)

    # split into train and validate
    index_train = np.nonzero(year_all < predict_year)[0]
    index_validate = np.nonzero(year_all == predict_year)[0]
    print 'train size', index_train.shape[0]
    print 'validate size', index_validate.shape[0]

    # calc train image mean (for each band), and then detract (broadcast)
    image_mean = np.mean(image_all[index_train], axis=(0, 1, 2))
    image_all = image_all - image_mean

    image_validate = image_all[index_validate]
    yield_validate = yield_all[index_validate]

    model = NeuralModel(name='net')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(
        tf.initialize_all_variables())  # DEPRECATED: use tf.global_variables_initializer - this is a required operation that handles all variable initialization.

    train_loss = 0
    val_loss = 0
    val_prediction = 0
    val_deviation = np.zeros([Config.B])
    # #########################
    # block when test
    # add saver
    saver = tf.train.Saver()
    summary_train_loss, summary_eval_loss, summary_RMSE, summary_ME = try_restore_model()

    RMSE_min = 100
    try:
        for i in range(Config.train_step):  # right now, 25000
            if i == 3500:
                Config.lr /= 10
                # summary_train_loss, summary_eval_loss, summary_RMSE, summary_ME = try_restore_model()
            if i == 20000:
                Config.lr /= 10
                # summary_train_loss, summary_eval_loss, summary_RMSE, summary_ME = try_restore_model()
            # if i==12000:
            #     Config.lr/=10
            # summary_train_loss, summary_eval_loss, summary_RMSE, summary_ME = try_restore_model()

            # No augmentation
            # index_train_batch = np.random.choice(index_train,size=Config.B)
            # image_train_batch = image_all[index_train_batch,:,0:Config.H,:]
            # yield_train_batch = yield_all[index_train_batch]
            # year_train_batch = year_all[index_train_batch,np.newaxis]

            # try data augmentation while training
            index_train_batch_1 = np.random.choice(index_train, size=Config.B)
            index_train_batch_2 = np.random.choice(index_train, size=Config.B)
            image_train_batch = (image_all[index_train_batch_1, :, 0:Config.H, :] + image_all[index_train_batch_1, :, 0:Config.H, :]) / 2  # supposed to use both right?
            yield_train_batch = (yield_all[index_train_batch_1] + yield_all[index_train_batch_1]) / 2
            # year_train_batch = (year_all[index_train_batch_1,np.newaxis]+year_all[index_train_batch_2,np.newaxis])/2

            index_validate_batch = np.random.choice(index_validate, size=Config.B)

            _, train_loss = sess.run([model.train_op, model.loss_err], feed_dict={
                model.x: image_train_batch,
                model.y: yield_train_batch,
                model.lr: Config.lr,
                model.keep_prob: Config.drop_out
            })

            if i % 200 == 0:
                val_loss, fc6, W, B = sess.run([model.loss_err, model.fc6, model.dense_W, model.dense_B], feed_dict={
                    model.x: image_all[index_validate_batch, :, 0:Config.H, :],
                    model.y: yield_all[index_validate_batch],
                    model.keep_prob: 1
                })

                print 'predict year %d step %d %f %f %f' % (predict_year, i, train_loss, val_loss, Config.lr)
                logging.info('predict year %d step %d %f %f %f', predict_year, i, train_loss, val_loss, Config.lr)

                # do validation
                pred = []
                real = []
                for j in range(image_validate.shape[0] / Config.B):
                    real_temp = yield_validate[j * Config.B:(j + 1) * Config.B]
                    pred_temp = sess.run(model.logits, feed_dict={
                        model.x: image_validate[j * Config.B:(j + 1) * Config.B, :, 0:Config.H, :],
                        model.y: yield_validate[j * Config.B:(j + 1) * Config.B],
                        model.keep_prob: 1
                    })
                    pred.append(pred_temp)
                    real.append(real_temp)
                pred = np.concatenate(pred)
                real = np.concatenate(real)
                RMSE = np.sqrt(np.mean((pred - real) ** 2))
                ME = np.mean(pred - real)

                if RMSE < RMSE_min:
                    RMSE_min = RMSE
                    # # save
                    # save_path = saver.save(sess, Config.save_path + str(predict_year)+'CNN_model.ckpt')
                    # print('save in file: %s' % save_path)
                    # np.savez(Config.save_path+str(predict_year)+'result.npz',
                    #     summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                    #     summary_RMSE=summary_RMSE,summary_ME=summary_RMSE)

                print 'Validation set', 'RMSE', RMSE, 'ME', ME, 'RMSE_min', RMSE_min
                logging.info('Validation set RMSE %f ME %f RMSE_min %f', RMSE, ME, RMSE_min)

                summary_train_loss.append(train_loss)
                summary_eval_loss.append(val_loss)
                summary_RMSE.append(RMSE)
                summary_ME.append(ME)

    except KeyboardInterrupt:
        print 'stopped'

    finally:
        # save
        save_path = saver.save(sess, Config.save_path + str(predict_year) + 'CNN_model.ckpt')
        print('save in file: %s' % save_path)
        logging.info('save in file: %s' % save_path)

        # save result
        pred_out = []
        real_out = []
        feature_out = []
        year_out = []
        locations_out = []
        index_out = []
        for i in range(image_all.shape[0] / Config.B):
            feature, pred = sess.run(
                [model.fc6, model.logits], feed_dict={
                    model.x: image_all[i * Config.B:(i + 1) * Config.B, :, 0:Config.H, :],
                    model.y: yield_all[i * Config.B:(i + 1) * Config.B],
                    model.keep_prob: 1
                })
            real = yield_all[i * Config.B:(i + 1) * Config.B]

            pred_out.append(pred)
            real_out.append(real)
            feature_out.append(feature)
            year_out.append(year_all[i * Config.B:(i + 1) * Config.B])
            locations_out.append(locations_all[i * Config.B:(i + 1) * Config.B])
            index_out.append(index_all[i * Config.B:(i + 1) * Config.B])
            # print i
        weight_out, b_out = sess.run(
            [model.dense_W, model.dense_B], feed_dict={
                model.x: image_all[0 * Config.B:(0 + 1) * Config.B, :, 0:Config.H, :],
                model.y: yield_all[0 * Config.B:(0 + 1) * Config.B],
                model.keep_prob: 1
            })
        pred_out = np.concatenate(pred_out)
        real_out = np.concatenate(real_out)
        feature_out = np.concatenate(feature_out)
        year_out = np.concatenate(year_out)
        locations_out = np.concatenate(locations_out)
        index_out = np.concatenate(index_out)

        path = Config.save_path + str(predict_year) + 'result_prediction.npz'
        np.savez(path,
                 pred_out=pred_out, real_out=real_out, feature_out=feature_out,
                 year_out=year_out, locations_out=locations_out, weight_out=weight_out, b_out=b_out,
                 index_out=index_out)

        # RMSE_GP,ME_GP,Average_GP=GaussianProcess(predict_year,path)
        # print 'RMSE_GP',RMSE_GP
        # print 'ME_GP',ME_GP
        # print 'Average_GP',Average_GP

        np.savez(Config.save_path + str(predict_year) + 'result.npz',
                 summary_train_loss=summary_train_loss, summary_eval_loss=summary_eval_loss,
                 summary_RMSE=summary_RMSE, summary_ME=summary_ME)

        plot_result()
