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
    return loss_cum + 50*loss_chi + 10*loss_cont


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
    return K.sum(K.square(y_pred[:, 200:-200] - y_pred[:, 199:-201]), axis=-1)


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
    cumsum_true = K.cumsum(y_true[:, 200:-200], axis=-1)
    cumsum_pred = K.cumsum(y_pred[:, 200:-200], axis=-1)
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
        K.square(y_true[:, 200:-200] - y_pred[:, 200:-200]),
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


def generator_rnn(rnn_input_size=30, rnn_output_size=8, **kwargs):
    gen_nsb = generator_nsb(**kwargs)
    for waveform_batch, proba_batch in gen_nsb:
        batch_size, num_sample = waveform_batch.shape
        num_input_per_waveform = num_sample - rnn_input_size + 1
        waveform_batch_rnn = np.zeros(
            [batch_size, num_input_per_waveform, rnn_input_size]
        )
        proba_batch_rnn = np.zeros(
            [batch_size, num_input_per_waveform, rnn_output_size]
        )
        for iteration in range(num_input_per_waveform):
            indexes_wf = range(iteration, iteration + rnn_input_size)
            waveform_batch_rnn[:, iteration, :] = waveform_batch[:, indexes_wf]
            indexes_pb = range(
                iteration * rnn_output_size,
                (iteration + 1) * rnn_output_size
            )
            proba_batch_rnn[:, iteration, :] = proba_batch[:, indexes_pb]
        yield waveform_batch_rnn, proba_batch_rnn


def loss_chi2_cnn(y_true, y_pred):
    """
    Function to calculate the chi2 loss. Compatible with
    tf.keras.Model.compile(). That loss is calculated as the sum of the squared
    difference of the prediction and the true number of photo-electrons.
    :param y_true: True value of the number of p.e. per bin of time.
    Its shape is (B, S, T) where B is the size of the batch S is the number of shift and
    T is the number of time bins.
    :param y_pred: Predicted probability of p.e. per bin of time.
    Its has the same shape as y_true.
    :return: the loss of each element in the batch. Its shape is (B, S)
    """
    loss_chi = K.sum(
        K.square(y_true - y_pred),
        axis=-1
    )
    return loss_chi


def continue_train_cnn(run_name):
    # toy parameters
    n_sample_init = 20
    pe_rate_mhz = (0, 200)
    bin_size_ns = 0.5
    sampling_rate_mhz = 250
    amplitude_gain = 5.
    noise_lsb = (0.5, 1.5)  # 1.05
    sigma_smooth_pe_ns = 1.
    baseline = 0
    relative_gain_std = 0.1

    # training parameters
    steps_per_epoch = 1e2  # 1 step feed a batch of events
    batch_size = 400  # number of waveform per batch
    epochs = 5
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
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        './Model/' + run_name + '.h5', verbose=1)
    # data generation for training
    generator = generator_nsb(
        n_event=1, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, pe_rate_mhz=pe_rate_mhz,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
        sigma_smooth_pe_ns=sigma_smooth_pe_ns, baseline=baseline,
        relative_gain_std=relative_gain_std
    )
    # training
    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[tb_cb, cp_callback],
    )
    model.save('./Model/' + run_name + 'r.h5')
    print('done training ' + run_name + 'r')


def train_rnn(lr=5e-4, n_sample_init=50, batch_size=10, shift_proba_bin=64,
              sigma_smooth_pe_ns=2.):

    initializer = tf.keras.initializers.Orthogonal()
    # model definition
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((4320, 1, 1), input_shape=(4320,)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(32, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.UpSampling2D(size=(2, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(64, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.UpSampling2D(size=(2, 1)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(128, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.UpSampling2D(size=(2, 1)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(256, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(128, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(64, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(32, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(8, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", kernel_initializer=initializer),
        tf.keras.layers.ReLU(
            negative_slope=1e-6, threshold=0, trainable=False
        ),
        tf.keras.layers.Flatten(),
    ])
    run_name = 'C32x16_U2_C64x16_U2_C128x8_U2_C256x8_C128x4_C64x4_C32x2_C8x2_C1x1_C1x1_ns0.1_shift' + str(shift_proba_bin) + '_all1-50-10lr' + str(lr) + "smooth" + str(sigma_smooth_pe_ns) + '_amsgrad'
    n_sample = model.input_shape[1]

    # model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, amsgrad=True),
        loss=loss_all,  # loss_all
        metrics=[loss_cumulative, loss_chi2, loss_continuity]  # loss_cumulative, loss_chi2, loss_continuity
    )
    print("number of parameters:", model.count_params())

    # data generation for training
    generator = generator_nsb(
        n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, pe_rate_mhz=(5, 400),
        bin_size_ns=0.5, sampling_rate_mhz=250,
        amplitude_gain=5., noise_lsb=(0.5, 1.5),
        sigma_smooth_pe_ns=sigma_smooth_pe_ns, baseline=0,
        relative_gain_std=0.05, shift_proba_bin=shift_proba_bin, dtype=np.float64
    )

    # training
    run = 0
    while os.path.exists('./Graph/' + run_name + '_run' + str(run)):
        run += 1
    run_name += '_run' + str(run)

    tbCallBack = tf.keras.callbacks.TensorBoard(
        log_dir='./Graph/' + run_name ,
        batch_size=batch_size
    )
    print()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        './Model/' + run_name + '.h5', verbose=1)

    steps_per_epoch = 200  # 1 step feed a batch of events
    epochs = 100

    try:
        model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[tbCallBack, cp_callback],
        )
    finally:
        # model.save('./Model/' + run_name + '.h5')  # done by ModelCheckpoint callback
        print('done training ' + run_name)
        return run_name


if __name__ == '__main__':
    from pe_extractor.plot_cnn import demo_nsb
    # tf.enable_eager_execution()
    shift = 64
    sigma_smooth_pe_ns = 1

    model_name = train_rnn(
        lr=0.0002, sigma_smooth_pe_ns=sigma_smooth_pe_ns,
        shift_proba_bin=shift, batch_size=30
    )
    #model_name = 'U2_C16x16_U2_C32x16_U2_C64x8_C32x8_C16x4_C8x4_C4x2_C2x2_C1x1_C1x1_ns0.1_shift32_both100lr0.0002smooth2_run0'
    demo_nsb(
        model_name, n_sample=4320, shift_proba_bin=shift, batch_index=0,
        sample_range=(3000, 3200), sigma_smooth_pe_ns=sigma_smooth_pe_ns
    )
    demo_nsb(
        model_name, n_sample=4320, shift_proba_bin=shift, batch_index=0,
        sample_range=(0, 4320), sigma_smooth_pe_ns=sigma_smooth_pe_ns
    )
    # continue_train_cnn('deconv_filters-16x20-8x10-4x10-2x10-1x1-1x1-1x1_lr0.0003_rel_gain_std0.1_pos_rate0-200_smooth1.0_noise0.5-1.5_baseline0_run0')
