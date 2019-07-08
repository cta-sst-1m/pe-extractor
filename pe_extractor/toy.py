import numpy as np
from scipy.interpolate import interp1d


def n_pe_from_rate(pe_rate_mhz, n_bin=800, bin_size_ns=0.5):
    """
    get number of pe per time bin
    :param pe_rate_mhz: pe rate in mhz
    :param n_bin: number of time bins to return
    :param bin_size_ns: duration of a time bin
    :
    :return: array of number of pe per time bin (size=(n_bin, batch_size))
    """
    rate_per_bin = pe_rate_mhz * bin_size_ns / 1000
    # the number of pes in a time bin follow a Poisson distribution
    n_pe_per_bin = np.random.poisson(rate_per_bin, n_bin)
    return n_pe_per_bin


def prepare_pulse_template(
        template_path='pulse_templates/SST-1M_01.txt', amplitude_gain=5., bin_size_ns=0.5,
        sampling_rate_mhz=250
):
    """
    function taking a template text file and returning an array ready for
    convolution.
    :param template_path: path to a text file where the 1st column is the time
    and the 2nd column is the amplitude of the pulse template.
    :param amplitude_gain: maximum amplitude of 1 p.e. in LSB
    :param bin_size_ns: duration of a time bin for the pes
    :param sampling_rate_mhz: sampling rate of the electronics
    :return: amplitude of the template ready for convolution.
    """
    pulse_template = np.loadtxt(template_path)[:, :2]
    pos_peak = np.argmax(pulse_template[:, 1])
    # make sure the amplitude correspond to the gain
    pulse_template[:, 1] *= amplitude_gain / pulse_template[pos_peak, 1]
    # add padding to the pulse so the start of the pulse is in the center
    peak_start = np.nonzero(pulse_template[:pos_peak, 1] < 1e-3)[0][-1] + 1
    pulse_template[:, 0] -= pulse_template[peak_start, 0]
    n_sample_after_start = len(pulse_template[(peak_start + 1):, 1])
    pulse_padding = np.zeros([n_sample_after_start, 2])
    pulse_padding[:, 0] = -pulse_template[-1:peak_start:-1, 0]
    pulse_template = np.vstack((pulse_padding, pulse_template[peak_start:, :2]))
    # normalize amplitude such that the convolution keeps the amplitude
    amplitude_norm = (1000 / sampling_rate_mhz / bin_size_ns)
    amplitude_pulse_template = pulse_template[:, 1] / amplitude_norm
    # interpolate the template to the pe binning
    t_pulse_template = pulse_template[:, 0] - pulse_template[0, 0]
    if t_pulse_template[1]-t_pulse_template[0] != bin_size_ns:
        amplitude_interp = interp1d(
            t_pulse_template, amplitude_pulse_template, 'quadratic'
        )
        t_bins_template = np.arange(0, t_pulse_template[-1], bin_size_ns)
        template_amplitude_bin = amplitude_interp(t_bins_template)
    else:
        template_amplitude_bin = amplitude_pulse_template
    return template_amplitude_bin


def waveform_from_n_pe(
        n_pe_per_bin, template_amplitude_bin, n_bin_per_sample=8,
        noise_lsb=1.05, baseline=0, rel_gain_std=0.1
):
    """
    get waveform from an array containing the number of pes per time bin
    :param n_pe_per_bin: array containing the number of pes per time
    bin
    :param template_amplitude_bin: array containing pulse template ready to be
    convolved, f.e. from prepare_pulse_template()
    :param n_bin_per_sample: number of time bin (for pes) per sample
    :param noise_lsb: amplitude in lsb of the electronic noise
    :param baseline: constant value added to the waveform
    :param rel_gain_std: gain standard  deviation divided by the gain.
    :return: waveform corresponding to the n_pe_per_bin array
    """
    n_bin = len(n_pe_per_bin)
    amplitude_pe = n_pe_per_bin * (1 + rel_gain_std * np.random.randn(n_bin))
    # convolve the number of pe to the template to get the samples
    # (with the pe binning size)
    waveform_bin = np.convolve(amplitude_pe, template_amplitude_bin, 'same')
    # integrate waveform bin to get the wanted sampling rate
    n_sample = int(np.floor(n_bin / n_bin_per_sample))
    waveform_bin = waveform_bin[:n_sample*n_bin_per_sample]
    waveform = waveform_bin.reshape([n_sample, n_bin_per_sample]).sum(-1)
    # add noise
    waveform += noise_lsb * np.random.randn(n_sample) + baseline
    return waveform


def _get_batch_nsb(
        batch_size, n_sample, n_bin_per_sample, pe_rate_mhz, bin_size_ns,
        template_amplitude_bin, noise_lsb, n_sample_init, baseline, kernel=None,
        relative_gain_std=0.1
):
    n_bin = n_sample * n_bin_per_sample
    n_bin_init = n_sample_init * n_bin_per_sample
    n_pe_batch = np.zeros([batch_size, n_bin - n_bin_init])
    waveform_batch = np.zeros([batch_size, n_sample - n_sample_init])
    if np.size(pe_rate_mhz) == 1:
        pe_rate_batch = np.ones(batch_size) * pe_rate_mhz
    elif np.size(pe_rate_mhz) == 2:
        # rates are chosen from the range of pe rate
        rate_min = np.min(pe_rate_mhz)
        rate_max = np.max(pe_rate_mhz)
        length_interval = rate_max - rate_min
        pe_rate_batch = np.random.random(batch_size) * length_interval + \
                            rate_min
    else:
        raise NotImplementedError(
            "pe_rate_mhz must be of size 1 (for constant value) or "
            "2 (for a range of values)")
    if np.size(noise_lsb) == 1:
        noise_lsb_batch = np.ones(batch_size) * noise_lsb
    elif np.size(noise_lsb) == 2:
        # rates are chosen from the range of pe rate
        noise_min = np.min(noise_lsb)
        noise_max = np.max(noise_lsb)
        length_interval = noise_max - noise_min
        noise_lsb_batch = np.random.random(batch_size) * length_interval + \
                          noise_min
    else:
        raise NotImplementedError(
            "noise_lsb must be of size 1 (for constant value) or "
            "2 (for a range of values)")
    if np.size(baseline) == 1:
        baseline_batch = np.ones(batch_size) * baseline
    elif np.size(baseline) == 2:
        # rates are chosen from the range of pe rate
        baseline_min = np.min(baseline)
        baseline_max = np.max(baseline)
        length_interval = baseline_max - baseline_min
        baseline_batch = np.random.random(batch_size) * length_interval + \
                         baseline_min
    else:
        raise NotImplementedError(
            "baseline must be of size 1 (for constant value) or "
            "2 (for a range of values)")
    for b in range(batch_size):
        n_pe = n_pe_from_rate(
            pe_rate_mhz=pe_rate_batch[b], n_bin=n_bin,
            bin_size_ns=bin_size_ns
        )
        waveform = waveform_from_n_pe(
            n_pe, template_amplitude_bin,
            n_bin_per_sample=n_bin_per_sample,
            noise_lsb=noise_lsb_batch[b],
            baseline=baseline_batch[b],
            rel_gain_std=relative_gain_std
        )
        if kernel is None:
            n_pe_batch[b, :] = n_pe[n_bin_init:]
        else:
            n_pe_smooth = np.convolve(n_pe, kernel, 'same')
            n_pe_batch[b, :] = n_pe_smooth[n_bin_init:]
        waveform_batch[b, :] = waveform[n_sample_init:]
    return waveform_batch, n_pe_batch


def _get_batch_flash(
                batch_size, n_sample, bin_flash, n_pe_flash, n_bin_per_sample,
                template_amplitude_bin, noise_lsb
):
    n_bin = n_sample * n_bin_per_sample
    n_pe_batch = np.zeros([batch_size, n_bin])
    amplitude_min = np.min(n_pe_flash)
    amplitude_max = np.max(n_pe_flash)
    diff_ampltidute = amplitude_max - amplitude_min
    noise_min = np.min(noise_lsb)
    noise_max = np.max(noise_lsb)
    diff_noise = noise_max - noise_min
    noise_lsb_batch = np.random.random(batch_size) * diff_noise + noise_min
    pe_flash = amplitude_min + diff_ampltidute * np.random.random(batch_size)
    n_pe_batch[:, bin_flash] = pe_flash
    waveform_batch = np.zeros([batch_size, n_sample])
    for b in range(batch_size):
        waveform_batch[b, :] = waveform_from_n_pe(
            n_pe_batch[b, :], template_amplitude_bin,
            n_bin_per_sample=n_bin_per_sample, noise_lsb=noise_lsb_batch[b]
        )
    return waveform_batch, n_pe_batch


def gauss_kernel(bin_size_ns, sigma_ns):
    """
    create a centered gaussian kernel on +-5 sigmas.
    :param bin_size_ns: interval in nano-seconds beween two consecutive
    kernel elements.
    :param sigma_ns: width of the guassian in nano-seconds
    :return: an numpy array of floats with the number of elements to go
    from -5 sigma_ns to + 5 sigma_ns by steps of sigma_ns containing
    the amplitude of a gaussian centered on 0 and whose width is sigma_ns.
    """
    nbin_5sigma = int(np.round(sigma_ns * 5 / bin_size_ns))
    t_kernel = np.arange(-nbin_5sigma, nbin_5sigma + .1, 1) * bin_size_ns
    gauss_kernel = np.exp(-0.5 * (t_kernel/ sigma_ns) ** 2)
    gauss_kernel /= np.sum(gauss_kernel)
    return gauss_kernel


def generator_flash(
        n_event=None, batch_size=1, n_sample=90, bin_flash=80,
        n_pe_flash=(1, 100), bin_size_ns=0.5, sampling_rate_mhz=250,
        amplitude_gain=5., noise_lsb=1.05
):
    """
    Generator returning for each iteration a batch of waveforms and a batch of
    pes. N photo-electrons are created at the same time (at after bin_flash
    bins), N being taken randomly in the range defined by n_pe_flash for
    each events.
    :param n_event: number of event returned. If None the generator never stops
    to return events.
    :param batch_size: number of events in each batch.
    :param n_sample: number of samples to simulate.
    :param bin_flash: index of the bin where the flash's photo-electrons are.
    :param n_pe_flash: Tuple giving the range of flash amplitude in
    photo-electrons. Each flash has a random amplitude taken in that range.
    :param bin_size_ns: size of a bin in nanoseconds for the photo-electrons.
    :param sampling_rate_mhz: sampling rate for the waveforms array.
    :param amplitude_gain: amplitude of a 1.pe. peak in LSB.
    :param noise_lsb: amplitude of random noise to add to the waveforms.
    Can be a tuple, then for each event of all batches the rate is taken in
    the range given by the tuple.
    :return: a generator of tuple of 2 arrays at each iteration. First array is
    a batch of waveforms (amplitude of the sampled signal) and a batch of pes
    (number of pes per bin). For both, first dimension is the batch iteration,
    second is along time (bin or sample).
    """
    template_amplitude_bin = prepare_pulse_template(
        template_path='pulse_templates/SST-1M_01.txt',
        amplitude_gain=amplitude_gain,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz
    )
    sample_size_ns = 1000 / sampling_rate_mhz
    n_bin_per_sample = sample_size_ns / bin_size_ns
    if abs(n_bin_per_sample - int(n_bin_per_sample)) > 1e-6:
        raise RuntimeError('there must be an integer number of bin per sample')
    n_bin_per_sample = int(n_bin_per_sample)
    if n_event is None:
        while True:
            waveform_batch, n_pe_batch = _get_batch_flash(
                batch_size, n_sample, bin_flash, n_pe_flash, n_bin_per_sample,
                template_amplitude_bin, noise_lsb,
            )
            yield (waveform_batch, n_pe_batch)
    else:
        for event in range(n_event):
            waveform_batch, n_pe_batch = _get_batch_flash(
                batch_size, n_sample, bin_flash, n_pe_flash, n_bin_per_sample,
                template_amplitude_bin, noise_lsb,
            )
            yield (waveform_batch, n_pe_batch)


def generator_nsb(
        n_event=None, batch_size=1, n_sample=90, n_sample_init=20,
        pe_rate_mhz=100, bin_size_ns=0.5, sampling_rate_mhz=250,
        amplitude_gain=5., noise_lsb=1.05, sigma_smooth_pe_ns=0.,
        baseline=0, relative_gain_std=0.1
):
    """
    Generator returning for each iteration a batch of waveforms and a batch of
    pes. The MC simulation is done at each call, no repeating of events
    occurs.
    :param n_event: number of event returned. If None the generator never stops
    to return events.
    :param batch_size: number of events in each batch.
    :param n_sample: number of samples to simulate. Must be larger than
    n_sample_init. Only n_sample - n_sample_init samples are returned for each
    waveform.
    :param n_sample_init: number of samples to skip at the beginning to take
    into account for effect of photo-electrons before the start of the window.
    :param pe_rate_mhz: rate of pe to simulate. Can be a tuple, then
    for each event of all batches the rate is taken in the range given by the
    tuple.
    :param bin_size_ns: size of bins in nanoseconds for the photo-electrons.
    :param sampling_rate_mhz: sampling rate for the waveforms array.
    :param amplitude_gain: amplitude of a 1.pe. peak in LSB.
    :param noise_lsb: amplitude of random noise to add to the waveforms.
    Can be a tuple, then for each event of all batches the rate is taken in the
    range given by the tuple.
    :param sigma_smooth_pe_ns: width of the gaussian kernel to
    convolve with the batch of pes. Use to convert the position of pe
    to a probability of with sigma_smooth_pe_ns. No convolution is done if
    sigma_smooth_pe_ns <= 0.
    :param baseline: baseline to add to the waveforms.
    Can be a tuple, then for each event of all batches the baseline is taken
    in the range given by the tuple.
    :param rel_gain_std: gain standard  deviation divided by the gain.
    :return: a generator of tuple of 2 arrays at each iteration. First array is
    a batch of waveforms (amplitude of the sampled signal) and a batch of p.e.
    (number of pes per bin). For both, first dimension is the batch iteration,
    second is along time (bin or sample).
    """
    template_amplitude_bin = prepare_pulse_template(
        template_path='pulse_templates/SST-1M_01.txt', amplitude_gain=amplitude_gain,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz
    )
    sample_size_ns = 1000 / sampling_rate_mhz
    n_bin_per_sample = sample_size_ns / bin_size_ns
    if abs(n_bin_per_sample - int(n_bin_per_sample)) > 1e-6:
        raise RuntimeError('there must be an integer number of bin per sample')
    n_bin_per_sample = int(n_bin_per_sample)
    # prepare kernel
    kernel = None
    if sigma_smooth_pe_ns > 0:
        kernel = gauss_kernel(bin_size_ns=bin_size_ns, sigma_ns=sigma_smooth_pe_ns)
        print("gaussian kernel prepared, shape=", kernel.shape)
    if n_event is None:
        while True:
            waveform_batch, n_pe_batch = _get_batch_nsb(
                batch_size, n_sample, n_bin_per_sample, pe_rate_mhz,
                bin_size_ns, template_amplitude_bin, noise_lsb, n_sample_init,
                baseline, kernel=kernel,
                relative_gain_std=relative_gain_std
            )
            yield (waveform_batch, n_pe_batch)
    else:
        for event in range(n_event):
            waveform_batch, n_pe_batch = _get_batch_nsb(
                batch_size, n_sample, n_bin_per_sample, pe_rate_mhz,
                bin_size_ns, template_amplitude_bin, noise_lsb, n_sample_init,
                baseline, kernel=kernel,
                relative_gain_std=relative_gain_std
            )
            yield (waveform_batch, n_pe_batch)


def model_predict(model, big_input, max_sample=1e4, continuous_waveform=False,
                  skip_bins=56):
    """
    Do model prediction, reshaping the input as needed to feed the model.
    :param model: tf.keras.model object
    (typically from tf.keras.models.load_model() )
    :param big_input: data used to make predictions
    :param max_sample: maximum number of samples to be run together.
    If the input is bigger, prection will be done by parts
    :param continuous_waveform: if True, the different waveforms
    within the batch are traited as they are a single long waveform.
    :param skip_bins: number of bin at the begining and at the end
    of the prediction to discard
    :return:
    """
    initial_shape = big_input.shape
    if continuous_waveform:
        big_input = big_input.reshape[1, -1]
    n_sample_input = big_input.shape[1]
    n_sample_network_input = model.input_shape[1]
    n_bin_network_output = model.output_shape[1]
    if abs((n_bin_network_output / n_sample_network_input ) % 1) > 0.01:
        raise ValueError('model is not compatible with model_predict()',
                         'as there is not a integer number of output time bin',
                         'per number of input sample')
    nbin_per_sample = int(
        np.round(n_bin_network_output / n_sample_network_input)
    )
    skip_sample = int(np.ceil(skip_bins / nbin_per_sample))
    if skip_sample * nbin_per_sample != skip_bins:
        skip_bins = skip_sample * nbin_per_sample
        print('WARNING: skip_sample changed to', skip_bins,
              'to have a correspoding integer number of samples')
    #print('nbin_per_sample:', nbin_per_sample)
    n_batch_input = big_input.shape[0]
    n_batch_max_network = int(np.floor(max_sample / n_sample_network_input))
    big_output = np.zeros([n_batch_input, n_sample_input * nbin_per_sample])
    #print('n_batch_max_network:', n_batch_max_network)
    #print('n_batch_input:', n_batch_input)
    sub_batch_start_events = range(
        0,
        n_batch_input,
        n_batch_max_network
    )
    for sub_batch_start in sub_batch_start_events:
        sub_batch_stop = min(sub_batch_start+n_batch_max_network, n_batch_input)
        sub_batch_input = big_input[sub_batch_start:sub_batch_stop, :]
        #print('sub_batch_start:', sub_batch_start)
        #print('sub_batch_stop:', sub_batch_stop)
        #print('sub_batch_input.shape:', sub_batch_input.shape)
        n_sub_batch_input = sub_batch_input.shape[0]
        sub_batch_pred = np.zeros(
            [n_sub_batch_input, nbin_per_sample * n_sample_input]
        )
        sample_start_points = range(
            0,
            n_sample_input,
            n_sample_network_input - 2 * skip_sample
        )
        for sample_start in sample_start_points:
            sample_end = sample_start + n_sample_network_input
            #print('sample_start:', sample_start)
            #print('sample_end:', sample_end)
            #print('n_sample_network_input:', n_sample_network_input)
            if sample_end > n_sample_input:  # too small imput
                # print('WARNING: input too small ({} samples) for network with {} samples as input. Padding with 0s.'.format(n_sample_input-sample_start, n_sample_network_input))
                n_valid_sample = n_sample_input - sample_start
                input_network = np.zeros([n_sub_batch_input, n_sample_network_input])
                input_network[:, :n_valid_sample] = sub_batch_input[:, sample_start:]
            else:
                n_valid_sample = sample_end - sample_start
                input_network = sub_batch_input[:, sample_start:sample_end]
            #print('n_valid_sample:', n_valid_sample)
            #print('input_network.shape', input_network.shape)
            pred_network = model.predict(input_network)
            #print('pred_network.shape', pred_network.shape)
            bin_start = sample_start * nbin_per_sample
            n_valid_bin = n_valid_sample * nbin_per_sample
            #we discard prediction for the first and last skip_bins
            bin_start = bin_start + skip_bins
            n_valid_bin -= 2 * skip_bins
            bin_end = bin_start + n_valid_bin
            # print('bin_start:', bin_start, 'bin_end:', bin_end)
            sub_batch_pred[:, bin_start:bin_end] = pred_network[:, skip_bins:n_valid_bin+skip_bins]
            #print('n_valid_bin:', n_valid_bin)
        big_output[sub_batch_start:sub_batch_stop, :] = sub_batch_pred
    output = big_output.reshape([initial_shape[0], initial_shape[1] * nbin_per_sample])
    return output


def generator_andrii_toy(filename, batch_size=1, n_sample=90, n_bin_per_sample=8):
    import ROOT
    import root_numpy

    f = ROOT.TFile.Open(filename)
    tree = f.Get("A")
    current_event = 0
    while True:
        n_pe = np.stack(root_numpy.tree2array(
            tree, "Npe",
            start=current_event, stop=current_event+batch_size
        ))[:, :n_sample]
        waveform = np.stack(root_numpy.tree2array(
            tree, "WaveformAmpliDetected",
            start=current_event, stop=current_event + batch_size
        ))[:, :n_sample]
        n_pe_upsampled = np.repeat(
            n_pe/n_bin_per_sample,
            n_bin_per_sample,
            axis=1
        )
        current_event += batch_size
        yield (waveform, n_pe_upsampled)


def plot_example(
        n_event=1, batch_size=1, n_sample=90, n_sample_init=20,
        pe_rate_mhz=100, bin_size_ns=0.5, sampling_rate_mhz=250,
        amplitude_gain=5., noise_lsb=1.05, baseline=0,
        relative_gain_std=0.1
):
    from matplotlib import pyplot as plt

    generator = generator_nsb(
        n_event=n_event, batch_size=batch_size, n_sample=n_sample,
        n_sample_init=n_sample_init,
        pe_rate_mhz=pe_rate_mhz, bin_size_ns=bin_size_ns,
        sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb, baseline=baseline,
        relative_gain_std=relative_gain_std
    )
    plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for waveform_batch, n_pe_batch in generator:
        for waveform, n_pe_per_bin in zip(waveform_batch, n_pe_batch):
            t_max_ns = len(n_pe_per_bin) * bin_size_ns
            t_bin = np.arange(0, t_max_ns, bin_size_ns)
            t_waveform = np.arange(0, t_max_ns, 1000 / sampling_rate_mhz)
            ax1.plot(t_bin, n_pe_per_bin)
            ax2.plot(t_waveform, waveform)
    plt.show()


def _read_experimental(rootfile, start=None, stop=None, step=None):
    import root_numpy

    wf0 = root_numpy.root2array(rootfile, "waveforms", "wf1",
                                start=start, stop=stop, step=step)
    wf1 = root_numpy.root2array(rootfile, "waveforms", "wf2",
                                start=start, stop=stop, step=step)
    t = None
    try:
        t = root_numpy.root2array(rootfile, "waveforms", "time",
                                    start=start, stop=stop, step=step)
    except ValueError:
        pass
    return wf0, wf1, t


def read_experimental(rootfiles, start=None, stop=None, step=None):
    if isinstance(rootfiles, str):
        wf0, wf1, t = _read_experimental(
            rootfiles, start=start, stop=stop, step=step
        )
        if wf0.ndim == 1 or wf1.ndim == 1:
            wf0 = wf0.reshape([1, -1])
            wf1 = wf1.reshape([1, -1])
        return wf0, wf1, t
    else:
        wf0_list = []
        wf1_list = []
        t_list = []
        for rootfile in rootfiles:
            wf0, wf1, t = _read_experimental(
                rootfile, start=start, stop=stop, step=step
            )
            wf0_list.append(wf0)
            wf1_list.append(wf1)
            t_list.append(t)
        return np.stack(wf0_list), np.stack(wf1_list), np.array(t_list)


def plot_waveforms(*waveforms):
    from matplotlib import pyplot as plt

    n_wf = len(waveforms)
    fig, axes = plt.subplots(n_wf, 1, sharex=True)
    for i, wf in enumerate(waveforms):
        axes[i].plot(waveforms[i].T)
    plt.show()
    plt.close(fig)


def hist_waveforms(*waveforms):
    from matplotlib import pyplot as plt

    n_wf = len(waveforms)
    fig, axes = plt.subplots(n_wf, 1, sharex=True)
    for i, wf in enumerate(waveforms):
        axes[i].hist(
            waveforms[i].T,
            np.arange(np.min(waveforms[i])-1, np.max(waveforms[i])+1, 1)
        )
        axes[i].set_yscale('log')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    #plot_example(n_event=3, batch_size=2, n_sample=90, n_sample_init=20, pe_rate_mhz=20, bin_size_ns=0.5, sampling_rate_mhz=250, amplitude_gain=5., noise_lsb=1.05, baseline=0, relative_gain_std=0.1)
    # wf0_hvoff, wf1_hvoff = read_experimental(
    #     'experimental_waveforms/SST1M_01_20190523_0000_0000_raw_waveforms.root',
    #     start=None, stop=100, step=None
    # )
    # baseline_wf0 = np.mean(wf0_hvoff)
    # baseline_wf1 = np.mean(wf1_hvoff)
    wf0, wf1, t = read_experimental(
        'experimental_waveforms/SST1M_01_20190523_0104_0108_raw_waveforms.root',
        start=None, stop=100, step=None
    )
    hist_waveforms(wf0.flatten(), wf1.flatten())
    plot_waveforms(wf0, wf1)
    #generator = generator_andrii_toy('/home/yves/Downloads/2.8V_8e+07_Hz_Compenstion_Off_test.root')
    #waveform_batch, n_pe_batch = next(generator)
