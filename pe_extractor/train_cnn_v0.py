import os
import numpy as np
import tensorflow as tf
from pe_extractor.toy import generator_nsb
from keras import backend as K


def histogram(array_nd, n_bin):
    array = K.flatten(array_nd)
    length = K.shape(array)[0]
    min_value = K.min(array)
    max_value = K.max(array)
    bin_size = (max_value - min_value) / n_bin
    bins = K.arange(min_value, max_value + .5 * bin_size, bin_size)
    bins_tiled = K.tile(K.reshape(bins, [1, n_bin+1]), [length, 1])
    array_tiled = K.tile(K.reshape(array, [length, 1]), [1, n_bin])
    ge_low_bin = K.greater_equal(array_tiled, bins_tiled[:, :-1])
    ge_high_bin = K.less(array_tiled, bins_tiled[:, 1:])
    in_bin = K.all(K.stack([ge_low_bin, ge_high_bin], axis=-1), axis=-1)
    histo = K.sum(K.cast(in_bin, tf.int32), axis=[0, 1])
    return histo, bins


def loss_proba(y_true, y_pred):
    n_bin = 100
    proba_flatten = K.flatten(y_pred)
    length = K.shape(proba_flatten)[0]
    min_proba = K.min(proba_flatten)
    max_value = K.max(proba_flatten)
    bin_size = (max_value - min_proba) / n_bin
    bins = K.arange(min_proba, max_value + .5 * bin_size, bin_size)
    bins_tiled = K.tile(K.reshape(bins, [1, n_bin + 1]), [length, 1])
    proba_tiled = K.tile(K.reshape(y_pred, [length, 1]), [1, n_bin])
    ge_low_bin = K.greater_equal(proba_tiled, bins_tiled[:, :-1])
    ge_high_bin = K.less(proba_tiled, bins_tiled[:, 1:])
    in_bin = K.all(K.stack([ge_low_bin, ge_high_bin], axis=-1), axis=-1)
    usage_proba = K.sum(K.cast(in_bin, tf.int32), axis=[0, 1])
    pe_per_proba = K.sum(tf.boolean_mask(y_true, in_bin), axis=[0, 1])
    real_probability_pe = pe_per_proba/usage_proba
    err_real_probability_pe = pe_per_proba**.5/usage_proba
    proba_pred = 0.5*(bins[1:] + bins[:-1])
    return K.sum(K.square(
        (proba_pred - real_probability_pe) / err_real_probability_pe
    ))


def loss_all(y_true, y_pred):
    """
    Function to calculate the total training loss. Compatible with
    tf.keras.Model.compile(). The  total loss is calculated as:
    10*loss_cumulative() + loss_chi2() + loss_continuity().
    :param y_true: True value of the number of p.e. per bin of time.
    Its shape is (B, T) where B is the size of the batch and
    T is the number of time bins.
    :param y_pred: Predicted probability of p.e. per bin of time.
    Its has the same shape as y_true.
    :return: the loss of each element in the batch. Its shape is (B)
    """
    loss_cum = loss_cumulative(y_true, y_pred)
    loss_chi = loss_chi2(y_true, y_pred)
    loss_cont = loss_continuity(y_true, y_pred)
    return 10 * loss_cum + loss_chi + loss_cont


def loss_continuity(_, y_pred):
    """
    Function to calculate the continuity loss. Compatible with
    tf.keras.Model.compile(). That loss is calculated as the sum of the squared
    difference of the prediction along 2 consecutive bins.
    :param _: Unused.
    :param y_pred: Predicted probability of p.e. per bin of time.
    Its has the same shape as y_true.
    :return: the loss of each element in the batch. Its shape is (B)
    """
    return K.sum(K.square(y_pred[:, 50:-50] - y_pred[:, 49:-51]), axis=-1)


def loss_cumulative(y_true, y_pred):
    """
    Function to calculate the cumulative loss. Compatible with
    tf.keras.Model.compile(). That loss is calculated as the averaged squared
    difference of the integrated prediction and the integrated true number of
    photo-electrons.
    :param y_true: True value of the number of p.e. per bin of time.
    Its shape is (B, T) where B is the size of the batch and
    T is the number of time bins.
    :param y_pred: Predicted probability of p.e. per bin of time.
    Its has the same shape as y_true.
    :return: the loss of each element in the batch. Its shape is (B)
    """
    cumsum_true = K.cumsum(y_true[:, 50:-50], axis=-1)
    cumsum_pred = K.cumsum(y_pred[:, 50:-50], axis=-1)
    loss_cum = K.mean(K.square(cumsum_pred - cumsum_true), axis=-1)
    return loss_cum


def loss_chi2(y_true, y_pred):
    """
    Function to calculate the chi2 loss. Compatible with
    tf.keras.Model.compile(). That loss is calculated as the sum of the squared
    difference of the prediction and the true number of photo-electrons.
    :param y_true: True value of the number of p.e. per bin of time.
    Its shape is (B, T) where B is the size of the batch and
    T is the number of time bins.
    :param y_pred: Predicted probability of p.e. per bin of time.
    Its has the same shape as y_true.
    :return: the loss of each element in the batch. Its shape is (B)
    """
    loss_chi = K.sum(
        K.square(y_true[:, 50:-50] - y_pred[:, 50:-50]),
        axis=-1
    )
    return loss_chi


def timebin_from_prediction(y_pred):
    """
    Function to transform the prediction (probabilty of getting a p.e per
    bin of time) to a discrete number of p.e. per bin of time.
    :param y_pred: Predicted probability of p.e. per bin of time.
    Its shape is (B, T) where B is the size of the batch and
    T is the number of time bins.
    :return:
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape([1, -1])
    cum_pe_prob = np.cumsum(y_pred, axis=-1)
    samples = np.arange(y_pred.shape[1])
    samples_bins = np.arange(y_pred.shape[1] + 1)
    timebin = np.zeros_like(y_pred)
    for b in range(y_pred.shape[0]):
        integer_n_pe = np.arange(0.5, np.round(cum_pe_prob[b, -1]) + 0.5)
        pe_samples = np.interp(integer_n_pe, cum_pe_prob[b, :], samples)
        timebin[b, :], _ = np.histogram(pe_samples, samples_bins)
    return timebin


def train_cnn():
    # training parameters
    steps_per_epoch = 100  # 1 step feed a batch of events
    batch_size = 400  # number of waveform per batch
    epochs = 10
    lr = 1e-3  # 1e-4

    # toy parameters
    n_sample_init = 20
    pe_rate_mhz = 0, 200
    bin_size_ns = 0.5
    sampling_rate_mhz = 250
    amplitude_gain = 5.
    noise_lsb = 0  # 1.05
    sigma_smooth_pe_ns = 0.25
    baseline = 0

    # model definition
    n_sample = 90
    n_filer1 = 4
    kernel_size = 10
    n_filer2 = 8
    n_filer3 = 8
    padding = "same"  # same
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
        tf.keras.layers.Flatten(),
        tf.keras.layers.ReLU(
            negative_slope=0, threshold=0, trainable=False
        )
    ])
    # model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=loss_all,  # loss_all, loss_chi2
        metrics=[loss_cumulative, loss_chi2, loss_continuity]  # 'accuracy'
    )
    print("number of parameters:", model.count_params())

    # data generation for training
    generator = generator_nsb(
        n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, pe_rate_mhz=pe_rate_mhz,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
        sigma_smooth_pe_ns=sigma_smooth_pe_ns, baseline=baseline
    )

    # training
    run = 0
    run_name = 'conv_filter' + str(n_filer1) + str(n_filer2) + str(n_filer3) + \
               '_kernel' + str(kernel_size) + '_lr' + str(lr)
    # run_name += '_dense'
    run_name += '_LCpos'
    if np.size(pe_rate_mhz) > 1:
        run_name += '_rate' + str(pe_rate_mhz[0]) + '-' + \
                    str(pe_rate_mhz[1])
    else:
        run_name += '_rate' + str(pe_rate_mhz)
    if sigma_smooth_pe_ns > 0:
        run_name += '_smooth' + str(sigma_smooth_pe_ns)
    if np.size(noise_lsb) > 1:
        run_name += '_noise' + str(noise_lsb[0]) + '-' + \
                    str(noise_lsb[1])
    else:
        run_name += '_noise' + str(noise_lsb)
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
    pe_rate_mhz = (0, 200)
    bin_size_ns = 0.5
    sampling_rate_mhz = 250
    amplitude_gain = 5.
    noise_lsb = 0  # 1.05
    sigma_smooth_pe_ns = 0.5

    # training parameters
    steps_per_epoch = 1e2  # 1 step feed a batch of events
    batch_size = 400  # number of waveform per batch
    epochs = 100
    # model
    model = tf.keras.models.load_model(
        './Model/' + run_name + '.h5',
        custom_objects={
            'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
            'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
        }
    )

    n_sample = model.input_shape[1]
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir='./Graph/' + run_name + 'r',
        batch_size=batch_size
    )
    # data generation for training
    generator = generator_nsb(
        n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, pe_rate_mhz=pe_rate_mhz,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
        sigma_smooth_pe_ns=sigma_smooth_pe_ns
    )
    # training
    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[tb_cb],
    )
    model.save('./Model/' + run_name + 'r.h5')
    print('done training ' + run_name + 'r')


if __name__ == '__main__':
    train_cnn()
    # continue_train_cnn('conv_filter488_kernel10_lr0.001_LC_rate0-200_smooth1_run2r')
