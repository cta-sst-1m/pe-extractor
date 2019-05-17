import os
import numpy as np
import tensorflow as tf
from toy import generator_for_training
from keras import backend as K


def loss_all(y_true, y_pred):
    return 10 * loss_cumulative(y_true, y_pred) + \
           loss_chi2(y_true, y_pred) + \
           loss_continuity(y_true, y_pred)


def loss_continuity(y_true, y_pred):
    return K.sum(K.square(y_pred[:, 50:-50] - y_pred[:, 49:-51]), axis=-1)


def loss_cumulative(y_true, y_pred):
    cumsum_true = K.cumsum(y_true[:, 50:-50], axis=-1)
    cumsum_pred = K.cumsum(y_pred[:, 50:-50], axis=-1)
    loss_cumulative = K.mean(K.square(cumsum_pred - cumsum_true), axis=-1)
    return loss_cumulative


def loss_chi2(y_true, y_pred):
    loss_chi2 = K.sum(K.square(y_true[:, 50:-50] - y_pred[:, 50:-50]), axis=-1)
    return loss_chi2


def timebin_from_prediction(y_pred):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape([1, -1])
    cum_photon_prob = np.cumsum(y_pred, axis=-1)
    samples = np.arange(y_pred.shape[1])
    samples_bins = np.arange(y_pred.shape[1] + 1)
    timebin = np.zeros_like(y_pred)
    for b in range(y_pred.shape[0]):
        integer_n_photon = np.arange(0.5, np.round(cum_photon_prob[b, -1]) + 0.5)
        photon_samples = np.interp(integer_n_photon, cum_photon_prob[b, :], samples)
        timebin[b, :], _ = np.histogram(photon_samples, samples_bins)
    return timebin


def train_cnn():
    # training parameters
    steps_per_epoch = 1e2  # 1 step feed a batch of events
    batch_size = 400  # number of waveform per batch
    epochs = 100
    lr = 1e-3  # 1e-4

    # toy parameters
    n_sample_init = 20
    photon_rate_mhz = 0, 200
    bin_size_ns = 0.5
    sampling_rate_mhz = 250
    amplitude_gain = 5.
    noise_lsb = 0  # 1.05
    sigma_smooth_photon_ns = 2

    # model definition
    n_sample = 90
    n_filer1 = 4
    kernel_size = 10
    n_filer2 = 8
    n_filer3 = 8
    padding = "same"  # causal
    n_bin = int(n_sample * 1000 / sampling_rate_mhz / bin_size_ns)

    model = tf.keras.Sequential([
        tf.keras.layers.Reshape(
            [n_sample, 1], input_shape=[n_sample], name="input_reshape"
        ),
        tf.keras.layers.Conv1D(
            filters=n_filer1, kernel_size=kernel_size, strides=1,
            padding=padding, name="conv1",
            activation="relu"
        ),
        tf.keras.layers.Conv1D(
            filters=n_filer2, kernel_size=kernel_size, strides=1,
            padding=padding, name="conv2",
            activation="relu"
        ),
        tf.keras.layers.Conv1D(
            filters=n_filer3, kernel_size=kernel_size, strides=1,
            padding=padding, name="conv3",
            activation="relu"
        ),
        tf.keras.layers.Reshape([n_sample * n_filer3, 1], name="reshape"),
        tf.keras.layers.ZeroPadding1D(4),
        tf.keras.layers.LocallyConnected1D(1, 9, name="LC"),
        # tf.keras.layers.Dense(units=n_bin, name="dense"),
        tf.keras.layers.Flatten()

    ])
    # model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=loss_all,  # loss_chi2
        metrics=[loss_cumulative, loss_chi2, loss_continuity]  # 'accuracy'
    )
    print("number of parameters:", model.count_params())

    # data generation for training
    generator = generator_for_training(
        n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, photon_rate_mhz=photon_rate_mhz,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
        sigma_smooth_photon_ns=sigma_smooth_photon_ns
    )

    # training
    run = 0
    run_name = 'conv_filter' + str(n_filer1) + str(n_filer2) + str(n_filer3) + \
               '_kernel' + str(kernel_size) + '_lr' + str(lr)
    #run_name += '_dense'
    run_name += '_LC'
    if np.size(photon_rate_mhz) > 1:
        run_name += '_rate' + str(photon_rate_mhz[0]) + '-' + \
                    str(photon_rate_mhz[1])
    else:
        run_name += '_rate' + str(photon_rate_mhz[0])
    if sigma_smooth_photon_ns > 0:
        run_name += '_smooth' + str(sigma_smooth_photon_ns)
    while os.path.exists('./Graph/' + run_name + '_run' + str(run)):
        run += 1

    tbCallBack = tf.keras.callbacks.TensorBoard(
        log_dir='./Graph/' + run_name + '_run' + str(run),
        batch_size=batch_size
    )

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[tbCallBack],
    )
    model.save('./Model/' + run_name + '_run' + str(run) + '.h5')
    print('done training ' + run_name + '_run' + str(run))


def continue_train_cnn(run_name):
    # toy parameters
    n_sample_init = 20
    photon_rate_mhz = (0, 200)
    bin_size_ns = 0.5
    sampling_rate_mhz = 250
    amplitude_gain = 5.
    noise_lsb = (0.5, 3)  # 1.05
    sigma_smooth_photon_ns = 2

    # training parameters
    steps_per_epoch = 1e2  # 1 step feed a batch of events
    batch_size = 400  # number of waveform per batch
    epochs = 300
    # model
    model = tf.keras.models.load_model(
        './Model/' + run_name + '.h5',
        custom_objects={
            'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
            'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
        }
    )

    n_sample = model.input_shape[1]
    tbCallBack = tf.keras.callbacks.TensorBoard(
        log_dir='./Graph/' + run_name + 'r',
        batch_size=batch_size
    )
    # data generation for training
    generator = generator_for_training(
        n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, photon_rate_mhz=photon_rate_mhz,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
        sigma_smooth_photon_ns=sigma_smooth_photon_ns
    )
    # training
    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[tbCallBack],
    )
    model.save('./Model/' + run_name + 'r.h5')
    print('done training ' + run_name + 'r')


if __name__ == '__main__':
    #train_cnn()
    continue_train_cnn('conv_filter488_kernel10_lr0.001_LC_rate0-200_smooth2_run0rrrr')

