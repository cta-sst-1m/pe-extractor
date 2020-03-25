import types
import tensorflow as tf
#tf.enable_eager_execution()
from pe_extractor.toy import generator_nsb, get_baseline, generator_andrii_toy, generator_andrii_toy_baselinesub
from keras import backend as K
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class Correlation:
    def __init__(
        self, shifts=tf.range(-100, 101), n_batch=10, n_sample=4320,
        sample_type=tf.int16, scope="wf", parallel_iterations=100
    ):
        with tf.name_scope(scope):
            self.shifts = tf.Variable(np.array(shifts), name="shift")
            self.wf1 = tf.placeholder(dtype=sample_type, shape=[n_batch, n_sample], name="wf1")
            self.wf2 = tf.placeholder(dtype=sample_type, shape=[n_batch, n_sample], name="wf2")
            self.sample_type = sample_type

            def correlate_with_delay(shift_sample):
                begin_wf1 = tf.stack(
                    [0, tf.maximum(shift_sample, 0)]
                )
                end_wf1 = tf.stack(
                    [n_batch, tf.minimum(n_sample+shift_sample, n_sample)]
                )
                size_wf1 = end_wf1 - begin_wf1
                wf1_shifted = tf.cast(
                    tf.slice(self.wf1, begin_wf1, size_wf1),
                    tf.float32
                )
                begin_wf2 = tf.stack([0, tf.maximum(-shift_sample, 0)])
                end_wf2 = tf.stack(
                    [n_batch, tf.minimum(n_sample-shift_sample, n_sample)]
                )
                size_wf2 = end_wf2 - begin_wf2
                wf2_shifted = tf.cast(
                    tf.slice(self.wf2, begin_wf2, size_wf2),
                    tf.float32
                )
                sum_1 = tf.reduce_sum(wf1_shifted)
                sum_2 = tf.reduce_sum(wf2_shifted)
                sum_11 = tf.reduce_sum(wf1_shifted*wf1_shifted)
                sum_12 = tf.reduce_sum(wf1_shifted*wf2_shifted)
                sum_22 = tf.reduce_sum(wf2_shifted*wf2_shifted)
                count = tf.size(wf1_shifted, out_type=tf.int64)
                return sum_1, sum_2, sum_11, sum_12, sum_22, count

            sum_1, sum_2, sum_11, sum_12, sum_22, count = tf.map_fn(
                correlate_with_delay,
                self.shifts,
                parallel_iterations=parallel_iterations,
                back_prop=False,
                dtype=(
                    tf.float32, tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.int64
                ),
                name="correlate_delays"
            )
            self.sum_1 = tf.identity(sum_1, name="sum_1")
            self.sum_2 = tf.identity(sum_2, name="sum_2")
            self.sum_11 = tf.identity(sum_11, name="sum_11")
            self.sum_12 = tf.identity(sum_12, name="sum_12")
            self.sum_22 = tf.identity(sum_22, name="sum_22")
            self.count = tf.identity(count, name="count")
            tf.Variable(0, name="dummy")
            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def __del__(self):
        self.sess.close()

    def __call__(self, wf1, wf2):
        """
        Function to calculate the correlatinos with batch of waveform as input as input
        :param wf1: batch of waveforms for the 1st pixel
        :param wf2: batch of waveforms for the 2nd pixel
        :return: sum_1, sum_2, sum_11, sum_12, sum_22, count: correlation for
        the shift in time-bins given at the constructor
        sum_1: float32 giving the sum of the 1st pixel samples
        sum_2: float32 giving the sum of the 2nd pixel samples
        sum_11: float32 giving the sum of the 1st pixel samples squared
        sum_12: float32 giving the sum of the product of the 1st and 2nd pixel samples
        sum_22: float32 giving the sum of the 2nd pixel samples squared
        count: int64 giving the number of samples
        """
        return self.sess.run(
            [
                self.sum_1, self.sum_2, self.sum_11,
                self.sum_12, self.sum_22, self.count
            ],
            feed_dict={self.wf1: wf1, self.wf2: wf2}
        )

    def generator(self, input_generator1, input_generator2):
        """
        Function to calculate the correlatinos with generator as input
        :param input_generator1: generator of batch of waveforms for the 1st pixel
        :param input_generator2: generator of batch of waveforms for the 2nd pixel
        :return: sum_1, sum_2, sum_11, sum_12, sum_22, count: correlation for
        the shift in time-bins given at the constructor
        sum_1: float32 giving the sum of the 1st pixel samples
        sum_2: float32 giving the sum of the 2nd pixel samples
        sum_11: float32 giving the sum of the 1st pixel samples squared
        sum_12: float32 giving the sum of the product of the 1st and 2nd pixel samples
        sum_22: float32 giving the sum of the 2nd pixel samples squared
        count: int64 giving the number of samples
        """
        sum_1, sum_2, sum_11, sum_12, sum_22, count = (None,) * 6
        for wf1, wf2 in tqdm(zip(input_generator1, input_generator2)):
            output = self.sess.run(
                [
                    self.sum_1, self.sum_2, self.sum_11,
                    self.sum_12, self.sum_22, self.count
                ],
                feed_dict={self.wf1: wf1, self.wf2: wf2}
            )
            if sum_1 is None:
                sum_1, sum_2, sum_11, sum_12, sum_22, count = output
            else:
                sum_1 += output[0]
                sum_2 += output[1]
                sum_11 += output[2]
                sum_12 += output[3]
                sum_22 += output[4]
                count += output[5]
        return sum_1, sum_2, sum_11, sum_12, sum_22, count

    def save(self, model_name="correlation"):
        from tensorflow.python.tools import freeze_graph

        with tf.get_default_graph().as_default() as g:
            saver=tf.train.Saver()
            print("save session")
            session_name = './models/' + model_name + '.ckpt'
            saver.save(self.sess, session_name)
            print("write graph")
            tf.train.write_graph(g, 'models/', model_name + '.pb', as_text=False)
            #tf.train.write_graph(g, 'models/', model_name + '.pbtxt', as_text=True)
        print("graph wrote")
        # we remove the last ":0" from the names
        name_sum1 = self.sum_1.name[:-2]
        name_sum2 = self.sum_2.name[:-2]
        name_sum11 = self.sum_11.name[:-2]
        name_sum12 = self.sum_12.name[:-2]
        name_sum22 = self.sum_22.name[:-2]
        name_nsample = self.count.name[:-2]
        print("freeze graph")
        frozen_name = 'models/' + model_name + '_frozen.pb'
        freeze_graph.freeze_graph(
            input_graph='models/' + model_name + '.pb',
            input_saver="",
            input_binary=True,
            input_checkpoint=session_name,
            output_node_names=name_sum1 + "," + name_sum2 + "," + name_sum11 + "," + name_sum12 + "," + name_sum22 + "," + name_nsample,
            restore_op_name="save/restore_all",
            filename_tensor_name="save/Const:0",
            output_graph=frozen_name,
            clear_devices=True,
            initializer_nodes="",
            variable_names_blacklist="dummy",
        )


class Extractor:
    """
    An Extractor object is used to process waveforms and get photo-electron
    probabilities.
    """
    orthogonal_initializer = tf.keras.initializers.Orthogonal()

    def __init__(self, n_sample, initializer=orthogonal_initializer):
        """
        Create an extractor object.
        :param n_sample: Number of samples in the waveforms
        :param initializer: A tf.keras.initializers , Orthogonal by default.
        """
        self.n_sample = n_sample
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape((n_sample, 1, 1), input_shape=(n_sample,)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(16, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.UpSampling2D(size=(2, 1)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(32, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.UpSampling2D(size=(2, 1)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=(64, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.UpSampling2D(size=(2, 1)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=(128, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.Conv2D(filters=4, kernel_size=(64, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.Conv2D(filters=4, kernel_size=(32, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.Conv2D(filters=2, kernel_size=(16, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.Conv2D(filters=2, kernel_size=(4, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(negative_slope=.1, max_value=None),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", kernel_initializer=initializer),
            tf.keras.layers.ReLU(
                negative_slope=1e-6, threshold=0, trainable=False
            ),
            tf.keras.layers.Flatten(),
        ])

    def train(
            self, run_name, lr=5e-4, n_sample_init=50, batch_size=10,
            shift_proba_bin=64, sigma_smooth_pe_ns=2., steps_per_epoch=200,
            epochs=100
    ):
        """
        train the CNN.
        :param run_name: name of the model
        :param lr: learning rate
        :param n_sample_init: parameter of the waveform generator used for
        training.
        :param batch_size: number of waveforms per batch
        :param shift_proba_bin: how many bins the photo-electron probabilities
        are shifted
        :param sigma_smooth_pe_ns: the pe truth (integers) is smoothed by a
        Gaussian of the given standard deviation. No smoothing is done if it is
        set to 0.
        :param steps_per_epoch: number of batch processed for each epoch
        :param epochs: number of epoch used in the training.
        """
        # model compilation
        print("compile model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr, amsgrad=True),
            loss=self.loss_all,
            metrics=[self.loss_cumulative, self.loss_chi2, self.loss_continuity]
        )
        print("model compiled, number of parameters:", self.model.count_params())

        # data generation for training
        generator = generator_nsb(
            n_event=None, batch_size=batch_size, n_sample=self.n_sample + n_sample_init,
            n_sample_init=n_sample_init, pe_rate_mhz=(5, 400),
            bin_size_ns=0.5, sampling_rate_mhz=250,
            amplitude_gain=5., noise_lsb=(0.5, 1.5),
            sigma_smooth_pe_ns=sigma_smooth_pe_ns, baseline=0,
            relative_gain_std=0.05, shift_proba_bin=shift_proba_bin, dtype=np.float64
        )

        #setting up callbacks
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='./Graph/' + run_name,
            batch_size=batch_size
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            './Model/' + run_name + '.h5', verbose=1)

        # training
        self.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[tb_callback, cp_callback],
        )

    def load(self, run_name):
        """
        Load an existing model
        :param run_name: name of the model
        """
        print("loading", run_name)
        model_loaded = tf.keras.models.load_model(
            './Model/' + run_name + '.h5',
            custom_objects={
                'loss_all': Extractor.loss_all,
                'loss_cumulative': Extractor.loss_cumulative,
                'loss_chi2': Extractor.loss_chi2,
                'loss_continuity': Extractor.loss_continuity
            }
        )
        self.model.set_weights(model_loaded.get_weights())

    def predict(self, wf):
        """
        extract photo-electrons from a batch of waveform
        :param wf: batch of waveform (shape BxS where B is the batch size and S
        is the number of sample per waveform)
        :return: a batch of photo-electron probabilities (shape BxP where P is
        the number of probability bins per waveforms)
        """
        return self.model.predict(wf)

    def predict_generator(self, generator):
        """
        extract photo-electrons from a generator
        :param generator: generator yielding lists of 2 elements where the first
        element is a batch of waveforms (shape BxS where B is the batch size
        and S is the number of sample per waveform)
        :return: generator yielding batches of photo-electron probabilities
        (shape BxP where P is the number of probability bins per waveforms)
        """
        for wf, _ in generator:
            pe_pred = self.model.predict(wf)
            yield pe_pred

    def predict_wf_generator(self, generator):
        """
        extract photo-electrons from a generator
        :param generator: generator yielding batches of waveforms
        (shape BxS where B is the batch size and S is the number of sample
        per waveform)
        :return: generator yielding batches of photo-electron probabilities
        (shape BxP where P is the number of probability bins per waveforms)
        """
        for wf in generator:
            pe_pred = self.model.predict(wf)
            yield pe_pred

    @staticmethod
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
        loss_cum = Extractor.loss_cumulative(y_true, y_pred)
        loss_chi = Extractor.loss_chi2(y_true, y_pred)
        loss_cont = Extractor.loss_continuity(y_true, y_pred)
        return loss_cum + 50*loss_chi + 10*loss_cont

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


def correlate_numpy(gen_wf1, gen_wf2, shifts):
    from pe_extractor.intensity_interferometry import calculate_correlation
    sum_1, sum_2, sum_11, sum_12, sum_22, count = (None,) * 6
    for wf1, wf2 in tqdm(zip(gen_wf1, gen_wf2)):
        output = calculate_correlation(
            wf1, wf2, shift_in_bins=shifts
        )
        if sum_1 is None:
            sum_1, sum_2, sum_11, sum_12, sum_22, count = output
        else:
            sum_1 += output[0]
            sum_2 += output[1]
            sum_11 += output[2]
            sum_12 += output[3]
            sum_22 += output[4]
            count += output[5]
    return sum_1, sum_2, sum_11, sum_12, sum_22, count


def plot_example_andrii_toy(
        model, datafile, n_sample, margin_lsb=8., samples_around=5,
        shift_proba_bin=64, xlim=None, index=0, batch_size=10, plot="show"
):
    assert index < batch_size
    extractor_toy_andrii = Extractor(n_sample)
    extractor_toy_andrii.load(model)
    wf, pe = next(generator_andrii_toy_baselinesub(
        datafile, batch_size=batch_size, n_sample=n_sample,
        margin_lsb=margin_lsb, samples_around=samples_around,
        shift_proba_bin=shift_proba_bin
    ))
    #pe_extract = extractor_toy_andrii.predict(wf)
    pe_extract = next(
        extractor_toy_andrii.predict_generator(
            generator_andrii_toy_baselinesub(
                datafile, batch_size=batch_size, n_sample=n_sample,
                margin_lsb=margin_lsb, samples_around=samples_around,
                shift_proba_bin=shift_proba_bin
            )
        )
    )
    del extractor_toy_andrii

    t_samples = np.arange(n_sample) * 4.
    t_bins = np.arange(n_sample*8) * .5
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex="col")
    if xlim is not None:
        axes[0].set_xlim(xlim)
    axes[0].plot(t_samples, wf[index, :])
    axes[0].set_ylabel("ADC value [LSB]")
    axes[1].plot(t_bins, pe[index, :], label='truth')
    axes[1].plot(t_bins, pe_extract[index, :], label='CNN')
    axes[1].set_ylabel("pe probability")
    axes[1].legend()
    cumsum_pe = np.cumsum(pe, axis=1)[index, :]
    cumsum_pe_extract = np.cumsum(pe_extract, axis=1)[index, :]
    axes[2].plot(t_bins, cumsum_pe, label='truth')
    axes[2].plot(t_bins, cumsum_pe_extract, label='CNN')
    axes[2].set_xlabel("time [ns]")
    axes[2].set_ylabel("cumulative # of pe")
    if xlim is None:
        is_in_xrange = range(n_sample*8)
    else:
        is_in_xrange = np.logical_and(xlim[0] < t_bins, t_bins < xlim[1])
    ymax = np.max(cumsum_pe[is_in_xrange])*1.1
    axes[2].set_ylim([-0.01, ymax])
    axes[2].legend()
    if plot == "show":
        plt.show()
    else:
        fig.savefig(plot)
        print(plot, " image created")


def generator_wf(generator_toy):
    for wf, _ in generator_toy:
        yield wf


def generator_pe_truth(generator_toy):
    for _, pe_truth in generator_toy:
        yield pe_truth


def g2_andrii_toy(
        model, datafile_pix1, datafile_pix2, batch_size=100, n_sample=2500,
        margin_lsb=8, samples_around=5, shifts=range(-500, 501),
        xlim=None, parallel_iterations=10,
        plot="show", n_bin_per_sample=8
):
    #get baselines
    gen_baseline_pix1 = generator_andrii_toy(
        datafile_pix1, batch_size=1000, n_sample=n_sample,
        n_bin_per_sample=n_bin_per_sample
    )
    waveforms_pix1, _ = next(gen_baseline_pix1)
    baseline_pix1 = get_baseline(
        waveforms_pix1, margin_lsb=margin_lsb, samples_around=samples_around
    )
    assert np.isfinite(baseline_pix1)
    del waveforms_pix1
    gen_baseline_pix2 = generator_andrii_toy(
        datafile_pix2, batch_size=1000, n_sample=n_sample,
        n_bin_per_sample=n_bin_per_sample
    )
    waveforms_pix2, _ = next(gen_baseline_pix2)
    baseline_pix2 = get_baseline(
        waveforms_pix2, margin_lsb=margin_lsb, samples_around=samples_around
    )
    assert np.isfinite(baseline_pix2)
    del waveforms_pix2

    # create correlator
    wf_correlation = Correlation(
        shifts=shifts, n_batch=batch_size,
        n_sample=n_sample, sample_type=tf.float32, scope="wf",
        parallel_iterations=parallel_iterations
    )
    # compute g2 for wf
    print("compute g2 on waveforms")
    gen_wf_pix1 = generator_wf(
        generator_andrii_toy(
            datafile_pix1, batch_size=batch_size, n_sample=n_sample,
            baseline=baseline_pix1
        )
    )
    gen_wf_pix2 = generator_wf(
        generator_andrii_toy(
            datafile_pix2, batch_size=batch_size, n_sample=n_sample,
            baseline=baseline_pix2
        )
    )
    sum_wf1, sum_wf2, _, sum_wf12, _, count_wf = wf_correlation(
        gen_wf_pix1, gen_wf_pix2
    )
    del wf_correlation

    # create correlator
    pe_correlation = Correlation(
        shifts=shifts, n_batch=batch_size,
        n_sample=n_sample * n_bin_per_sample, sample_type=tf.float32, scope="pe",
        parallel_iterations=parallel_iterations
    )

    # compute g2 for MC truth
    print("compute g2 on MC truth")
    gen_pe_truth_pix1 = generator_pe_truth(
        generator_andrii_toy(
            datafile_pix1, batch_size=batch_size, n_sample=n_sample,
            baseline=baseline_pix1
        )
    )
    gen_pe_truth_pix2 = generator_pe_truth(
        generator_andrii_toy(
            datafile_pix2, batch_size=batch_size, n_sample=n_sample,
            baseline=baseline_pix2
        )
    )
    sum_pe1_truth, sum_pe2_truth, _, sum_pe12_truth, _, count_pe_truth = pe_correlation(
        gen_pe_truth_pix1, gen_pe_truth_pix2
    )

    # create extractor
    extractor_toy_andrii = Extractor(n_sample)
    extractor_toy_andrii.load(model)

    # compute g2 for extracted pe
    print("compute g2 on extracted pe")
    gen_pe_pred_pix1 = extractor_toy_andrii.predict_generator(
        generator_andrii_toy(
            datafile_pix1, batch_size=batch_size, n_sample=n_sample,
            baseline=baseline_pix1
        )
    )
    gen_pe_pred_pix2 = extractor_toy_andrii.predict_generator(
        generator_andrii_toy(
            datafile_pix2, batch_size=batch_size, n_sample=n_sample,
            baseline=baseline_pix2
        )
    )
    sum_pe1, sum_pe2, _, sum_pe12, _, count_pe = pe_correlation(
        gen_pe_pred_pix1, gen_pe_pred_pix2
    )
    del extractor_toy_andrii
    del pe_correlation

    # plot
    print("plotting")
    n_sample_tot = count_wf[np.array(shifts) == 0][0]
    n_wf_tot = int(n_sample_tot / n_sample)
    time_tot_us = n_sample_tot * 4e-3
    title = "g2 with " + str(n_wf_tot) + " waveforms"
    title += " (" + str(time_tot_us*1e-3) + " ms)"
    fig = plt.figure(figsize=(8, 6))
    g2_wf = count_wf * sum_wf12 / (sum_wf1*sum_wf2)
    g2_pe = count_pe * sum_pe12 / (sum_pe1*sum_pe2)
    g2_pe_truth = count_pe_truth * sum_pe12_truth / (sum_pe1_truth*sum_pe2_truth)
    shift_wf_ns = 4. * np.array(shifts)
    shift_pe_ns = .5 * np.array(shifts)
    plt.plot(shift_wf_ns, g2_wf, '+-', label='g2 from waveforms')
    plt.plot(shift_pe_ns, g2_pe, '+-', label='g2 from CNN')
    plt.plot(shift_pe_ns, g2_pe_truth, '+-', label='g2 from MC truth')
    plt.xlabel('delay [ns]')
    plt.ylabel(r'$g^2(\tau)$')
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.grid()
    plt.title(title)
    if plot == "show":
        plt.show()
    else:
        fig.savefig(plot)
        print(plot, "image created")


def generator_rootdatafile(
        filename, batch_size=1, n_sample=4320, baseline=0,
        tree_name="waveforms", branch_name="wf1"
):
    import ROOT
    import root_numpy

    f = ROOT.TFile.Open(filename)
    tree = f.Get(tree_name)
    if tree.Class_Name() != "TTree" and tree.Class_Name() != "TNtuple":
        ValueError(filename + " has no tree " + tree_name)
    n_waveform = tree.GetEntries()
    pe1_branch = tree.GetBranch(branch_name)
    if pe1_branch.Class_Name() != "TBranch":
        ValueError("no branch " + branch_name + " in tree " + tree_name)
    current_event = 0
    while current_event + batch_size <= n_waveform:
        waveform = np.stack(root_numpy.tree2array(
            tree, "wf1",
            start=current_event, stop=current_event + batch_size
        ))[:, :n_sample]
        current_event += batch_size
        yield waveform - baseline


def generator_rootdatafile_baselinesub(
        filename, batch_size=1, n_sample=4320,
        tree_name="waveforms", branch_name="wf1",
        n_wf_baseline=1000, margin_lsb=8, samples_around=4
):
    gen_baseline = generator_rootdatafile(
        filename, batch_size=n_wf_baseline, n_sample=n_sample,
        tree_name=tree_name, branch_name=branch_name
    )
    waveforms = next(gen_baseline)
    baseline = get_baseline(
        waveforms, margin_lsb=margin_lsb, samples_around=samples_around
    )
    assert np.isfinite(baseline)
    del waveforms
    return generator_rootdatafile(
        filename, batch_size=batch_size, n_sample=n_sample,
        tree_name=tree_name, branch_name=branch_name, baseline=baseline
    )


def plot_example_data(
        model, datafile, n_sample=4320, margin_lsb=8., samples_around=5,
        xlim=None, index=0, batch_size=10, plot="show", branch_name="wf1"
):
    from matplotlib import pyplot as plt

    assert index < batch_size
    extractor_data = Extractor(n_sample)
    extractor_data.load(model)
    wf = next(generator_rootdatafile_baselinesub(
        datafile, batch_size=batch_size, n_sample=n_sample,
        tree_name="waveforms", branch_name=branch_name, n_wf_baseline=1000,
        margin_lsb=margin_lsb, samples_around=samples_around,
    ))
    pe_extract = next(
        extractor_data.predict_wf_generator(
            generator_rootdatafile_baselinesub(
                datafile, batch_size=batch_size, n_sample=n_sample,
                tree_name="waveforms", branch_name=branch_name, n_wf_baseline=1000,
                margin_lsb=margin_lsb, samples_around=samples_around,
            )
        )
    )
    del extractor_data

    t_samples = np.arange(n_sample) * 4.
    t_bins = np.arange(n_sample*8) * .5
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex="col")
    if xlim is not None:
        axes[0].set_xlim(xlim)
    axes[0].plot(t_samples, wf[index, :])
    axes[0].set_ylabel("ADC value [LSB]")
    axes[1].plot(t_bins, pe_extract[index, :], label='CNN')
    axes[1].set_ylabel("pe probability")
    axes[1].legend()
    cumsum_pe_extract = np.cumsum(pe_extract, axis=1)[index, :]
    axes[2].plot(t_bins, cumsum_pe_extract, label='CNN')
    axes[2].set_xlabel("time [ns]")
    axes[2].set_ylabel("cumulative # of pe")
    if xlim is None:
        is_in_xrange = range(n_sample*8)
    else:
        is_in_xrange = np.logical_and(xlim[0] < t_bins, t_bins < xlim[1])
    ymax = np.max(cumsum_pe_extract[is_in_xrange])*1.1
    axes[2].set_ylim([-0.01, ymax])
    axes[2].legend()
    if plot == "show":
        plt.show()
    else:
        fig.savefig(plot)
        print(plot, " image created")


def g2_wf_data(
        root_datafile, batch_size=100, n_sample=4320,
        margin_lsb=8, samples_around=5, shifts=range(-500, 501),
        xlim=None, parallel_iterations=100, plot="show", method="tf"
):
    from matplotlib import pyplot as plt

    gen_baseline_pix1 = generator_rootdatafile(
        root_datafile, batch_size=1000, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf1"
    )
    waveforms_pix1 = next(gen_baseline_pix1)
    baseline_pix1 = get_baseline(
        waveforms_pix1, margin_lsb=margin_lsb, samples_around=samples_around
    )
    assert np.isfinite(baseline_pix1)
    del waveforms_pix1
    gen_baseline_pix2 = generator_rootdatafile(
        root_datafile, batch_size=1000, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf2"
    )
    waveforms_pix2 = next(gen_baseline_pix2)
    baseline_pix2 = get_baseline(
        waveforms_pix2, margin_lsb=margin_lsb, samples_around=samples_around
    )
    assert np.isfinite(baseline_pix2)
    del waveforms_pix2

    # compute g2 for wf
    print("compute g2 on waveforms")
    gen_wf_pix1 = generator_rootdatafile(
        root_datafile, batch_size=batch_size, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf1", baseline=baseline_pix1
    )
    gen_wf_pix2 = generator_rootdatafile(
        root_datafile, batch_size=batch_size, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf2", baseline=baseline_pix2
    )

    # create correlator
    if method == "tf":
        wf_correlation = Correlation(
            shifts=shifts, n_batch=batch_size,
            n_sample=n_sample, sample_type=tf.float32, scope="wf",
            parallel_iterations=parallel_iterations
        )
        sum_wf1, sum_wf2, _, sum_wf12, _, count_wf = wf_correlation(
            gen_wf_pix1, gen_wf_pix2
        )
        del wf_correlation
    elif method == "numpy":
        sum_wf1, sum_wf2, sum_wf12, _, _, count_wf = correlate_numpy(
            gen_wf_pix1, gen_wf_pix2, shifts=shifts
        )
    else:
        ValueError("method must be \"tf\" or \"numpy\"")
    # plot
    print("plotting")
    n_sample_tot = count_wf[np.array(shifts) == 0][0]
    n_wf_tot = int(n_sample_tot / n_sample)
    time_tot_us = n_sample_tot * 4e-3
    title = "g2 with {} waveforms ({:.3f} ms)".format(n_wf_tot, time_tot_us*1e-3)
    fig = plt.figure(figsize=(4, 3))
    g2_wf = count_wf * sum_wf12 / (sum_wf1*sum_wf2)
    shift_wf_ns = 4. * np.array(shifts)
    plt.plot(shift_wf_ns, g2_wf, label='g2 from waveforms')
    plt.xlabel('delay [ns]')
    plt.ylabel(r'$g^2(\tau)$')
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.tight_layout()
    if plot == "show":
        plt.show()
    else:
        fig.savefig(plot)
        print(plot, "image created")


def g2_data(
        model, root_datafile, batch_size=100, n_sample=4320,
        margin_lsb=8, samples_around=5, shifts=range(-500, 501),
        xlim=None, parallel_iterations=10,
        plot="show", g2_datafile=None, n_bin_per_sample=8
):
    from matplotlib import pyplot as plt

    #get baselines
    gen_baseline_pix1 = generator_rootdatafile(
        root_datafile, batch_size=10000, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf1"
    )
    waveforms_pix1 = next(gen_baseline_pix1)
    baseline_pix1 = get_baseline(
        waveforms_pix1, margin_lsb=margin_lsb, samples_around=samples_around
    )
    assert np.isfinite(baseline_pix1)
    del waveforms_pix1
    gen_baseline_pix2 = generator_rootdatafile(
        root_datafile, batch_size=1000, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf2"
    )
    waveforms_pix2 = next(gen_baseline_pix2)
    baseline_pix2 = get_baseline(
        waveforms_pix2, margin_lsb=margin_lsb, samples_around=samples_around
    )
    assert np.isfinite(baseline_pix2)
    del waveforms_pix2

    # create correlator
    wf_correlation = Correlation(
        shifts=shifts, n_batch=batch_size,
        n_sample=n_sample, sample_type=tf.float32, scope="wf",
        parallel_iterations=parallel_iterations
    )
    # compute g2 for wf
    print("compute g2 on waveforms")
    gen_wf_pix1 = generator_rootdatafile(
        root_datafile, batch_size=batch_size, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf1", baseline=baseline_pix1
    )
    gen_wf_pix2 = generator_rootdatafile(
        root_datafile, batch_size=batch_size, n_sample=n_sample,
        tree_name="waveforms", branch_name="wf2", baseline=baseline_pix2
    )
    sum_wf1, sum_wf2, sum_wf11, sum_wf12, sum_wf22, count_wf = wf_correlation(
        gen_wf_pix1, gen_wf_pix2
    )
    del wf_correlation

    # create correlator
    pe_correlation = Correlation(
        shifts=shifts, n_batch=batch_size,
        n_sample=n_sample * n_bin_per_sample, sample_type=tf.float32, scope="pe",
        parallel_iterations=parallel_iterations
    )

    sum_pe1, sum_pe2, sum_pe11, sum_pe12, sum_pe22, count_pe = None, None, None, None, None, None
    if model is not None:
        # create extractor
        extractor = Extractor(n_sample)
        extractor.load(model)

        # compute g2 for extracted pe
        print("compute g2 on extracted pe")
        gen_pe_pred_pix1 = extractor.predict_wf_generator(
            generator_rootdatafile(
                root_datafile, batch_size=batch_size, n_sample=n_sample,
                tree_name="waveforms", branch_name="wf1", baseline=baseline_pix1
            )
        )
        gen_pe_pred_pix2 = extractor.predict_wf_generator(
            generator_rootdatafile(
                root_datafile, batch_size=batch_size, n_sample=n_sample,
                tree_name="waveforms", branch_name="wf2", baseline=baseline_pix2
            )
        )
        sum_pe1, sum_pe2, sum_pe11, sum_pe12, sum_pe22, count_pe = pe_correlation(
            gen_pe_pred_pix1, gen_pe_pred_pix2
        )
        del extractor
        del pe_correlation

    if g2_datafile is not None:
        np.savez(
            g2_datafile, shift_in_sample=shifts,
            n_sample_wf=n_sample, sum1_wf=sum_wf1, sum2_wf=sum_wf2,
            sum12_wf=sum_wf12, sum11_wf=sum_wf11, sum22_wf=sum_wf22,
            shift_in_bins=shifts,
            n_sample_pb=n_sample *8, sum1_pb=sum_pe1, sum2_pb=sum_pe2,
            sum12_pb=sum_pe12, sum11_pb=sum_pe11, sum22_pb=sum_pe22,
            baseline_pix0=baseline_pix1, baseline_pix1=baseline_pix2,
        )

    # plot
    if plot is not None:
        print("plotting")
        n_sample_tot = count_wf[np.array(shifts) == 0][0]
        n_wf_tot = int(n_sample_tot / n_sample)
        time_tot_us = n_sample_tot * 4e-3
        title = "g2 with {} waveforms ({:.3f} ms)".format(n_wf_tot, time_tot_us*1e-3)
        fig = plt.figure(figsize=(8, 6))
        g2_wf = count_wf * sum_wf12 / (sum_wf1*sum_wf2)
        g2_pe = count_pe * sum_pe12 / (sum_pe1*sum_pe2)
        shift_wf_ns = 4. * np.array(shifts)
        shift_pe_ns = .5 * np.array(shifts)
        plt.plot(shift_wf_ns, g2_wf, label='g2 from waveforms')
        if model is not None:
            plt.plot(shift_pe_ns, g2_pe, label='g2 from CNN')
        plt.xlabel('delay [ns]')
        plt.ylabel(r'$g^2(\tau)$')
        if xlim is not None:
            plt.xlim(xlim)
        plt.legend()
        plt.grid()
        plt.title(title)
        plt.tight_layout()
        if plot == "show":
            plt.show()
        else:
            fig.savefig(plot)
            print(plot, "image created")


def charge_resolution(
        model, n_sample, bin_flash, n_pe_flashes, dcr_mhz=4., batch_size=10000,
        shift_proba_bin=-64
):
    from pe_extractor.toy import waveform_from_n_pe, n_pe_from_rate, prepare_pulse_template
    from matplotlib import pyplot as plt

    template_amplitude_bin = prepare_pulse_template(
        template_path='pulse_templates/SST-1M_01.txt',
        amplitude_gain=5.,
        bin_size_ns=0.5, sampling_rate_mhz=250
    )
    extractor = Extractor(n_sample)
    extractor.load(model)
    range_int = slice(bin_flash-3*8, bin_flash+5*8)
    mean_pe = np.zeros(len(n_pe_flashes))
    std_pe = np.zeros(len(n_pe_flashes))
    for i, n_pe_flash in tqdm(enumerate(n_pe_flashes)):
        waveform = np.zeros([batch_size, n_sample])
        for b in range(batch_size):
            n_pe_per_bin = n_pe_from_rate(
                pe_rate_mhz=dcr_mhz, n_bin=n_sample*8,
                bin_size_ns=0.5
            )
            n_pe_per_bin[bin_flash] += np.random.poisson(n_pe_flash)

            waveform[b, :] = waveform_from_n_pe(
                n_pe_per_bin, template_amplitude_bin, n_bin_per_sample=8,
                noise_lsb=1.05, baseline=0, rel_gain_std=0.1
            )
        pe = extractor.predict(waveform)
        pe_shifted = np.roll(pe, shift_proba_bin, axis=1)
        if shift_proba_bin > 0:
            pe_shifted[:, :shift_proba_bin] = 0
        elif shift_proba_bin < 0:
            pe_shifted[:, shift_proba_bin:] = 0
        pe_measured = np.sum(pe_shifted[:, range_int], axis=1)
        mean_pe[i] = np.mean(pe_measured)
        std_pe[i] = np.std(pe_measured)
    fig = plt.figure(figsize=(8,6))
    plt.errorbar(
        n_pe_flashes, mean_pe, xerr=np.sqrt(n_pe_flashes), yerr=std_pe,
        fmt='.'
    )
    plt.plot(n_pe_flashes, n_pe_flashes, 'k--')
    plt.xlabel('injected # of pe')
    plt.ylabel('reconstructed # of pe')
    plt.savefig('reconstructed_charge.png')
    plt.close(fig)

    fig = plt.figure(figsize=(8,6))
    plt.semilogx(n_pe_flashes, std_pe/mean_pe, label="resolution")
    plt.plot(n_pe_flashes, 1/np.sqrt(n_pe_flashes), label="Poisson limit")
    plt.xlabel('injected # of pe')
    plt.ylabel('charge resolution [1]')
    plt.legend()
    plt.savefig('charge_resolution.png')
    plt.close(fig)


if __name__ == '__main__':
    import time

    model = 'C16x16_U2_C32x16_U2_C64x8_U2_C128x8_C64x4_C32x4_C16x2_C4x2_C1x1_C1x1_ns0.1_shift64_all1-50-10lr0.0002smooth1_amsgrad_run0'
    # toy_datafiles = [
    #     'experimental_waveforms/CorrON40Mhz_NoNSB_NoNoise_PDE100_Ch1.root',
    #     'experimental_waveforms/CorrON40Mhz_NoNSB_NoNoise_PDE100_Ch2.root'
    # ]
    #
    # plot_example_andrii_toy(
    #     model, toy_datafiles[0], 2500, xlim=(0, 1000), margin_lsb=9,
    #     samples_around=5, index=9, plot='toy_reco_pix1.png'
    # )
    # plot_example_andrii_toy(
    #     model, toy_datafiles[1], 2500, xlim=(0, 1000), margin_lsb=9,
    #     samples_around=5, index=9, plot='toy_reco_pix2.png'
    # )

    # start = time.time()
    # g2_andrii_toy(
    #     model, toy_datafiles[0], toy_datafiles[1], batch_size=50,
    #     n_sample=2500, margin_lsb=9, samples_around=5,
    #     shifts=range(-200, 201), xlim=(-100, 100),
    #     parallel_iterations=48, plot="test.png"
    # )
    # end = time.time()
    # print("elapsed", end - start, "s")

    # datafile = 'experimental_waveforms/SST1M_01_20190917_0197_0197_raw_waveforms.root'
    # plot_example_data(
    #     model, datafile, 2500, xlim=(0, 1000), margin_lsb=9,
    #     samples_around=5, index=9, plot='sp_reco_pix1.png', branch_name="wf1"
    # )

    charge_resolution(
        model, n_sample=90, bin_flash=20*8,
        n_pe_flashes=np.logspace(0, 3, 20), dcr_mhz=4.
    )

    # start = time.time()
    # g2_data(
    #     model, datafile, batch_size=50,
    #     n_sample=4320, margin_lsb=9, samples_around=5,
    #     shifts=range(-200, 201), xlim=(-100, 100),
    #     parallel_iterations=48, plot="g2_off.png"
    # )
    # g2_wf_data(
    #     datafile, batch_size=200,
    #     n_sample=4320, margin_lsb=9, samples_around=5,
    #     shifts=range(-250, 251), xlim=(-1000, 1000),
    #     parallel_iterations=48, method="tf",
    #     plot="g2_wf_pp.png"
    # )
    # end = time.time()
    # print("elapsed", end - start, "s")