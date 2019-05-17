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
        noise_lsb=1.05
):
    """
    get waveform from an array containing the number of pes per time bin
    :param n_pe_per_bin: array containing the number of pes per time
    bin
    :param template_amplitude_bin: array containing pulse template ready to be
    convolved, f.e. from prepare_pulse_template()
    :param n_bin_per_sample: number of time bin (for pes) per sample
    :param noise_lsb: amplitude in lsb of the electronic noise
    :return: waveform corresponding to the n_pe_per_bin array
    """
    # convolve the number of pe to the template to get the samples
    # (with the pe binning size)
    waveform_bin = np.convolve(n_pe_per_bin, template_amplitude_bin, 'same')
    # integrate waveform bin to get the wanted sampling rate
    n_sample = int(np.floor(len(n_pe_per_bin) / n_bin_per_sample))
    waveform_bin = waveform_bin[:n_sample*n_bin_per_sample]
    waveform = waveform_bin.reshape([n_sample, n_bin_per_sample]).sum(-1)
    # add noise
    waveform += noise_lsb * np.random.randn(n_sample)
    return waveform


def _get_batch(
        batch_size, n_sample, n_bin_per_sample, pe_rate_mhz, bin_size_ns,
        template_amplitude_bin, noise_lsb, n_sample_init, kernel=None
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
    for b in range(batch_size):
        n_pe = n_pe_from_rate(
            pe_rate_mhz=pe_rate_batch[b], n_bin=n_bin,
            bin_size_ns=bin_size_ns
        )
        waveform = waveform_from_n_pe(
            n_pe, template_amplitude_bin,
            n_bin_per_sample=n_bin_per_sample, noise_lsb=noise_lsb_batch[b]
        )
        if kernel is None:
            n_pe_batch[b, :] = n_pe[n_bin_init:]
        else:
            n_pe_smooth = np.convolve(n_pe, kernel, 'same')
            n_pe_batch[b, :] = n_pe_smooth[n_bin_init:]
        waveform_batch[b, :] = waveform[n_sample_init:]
    return waveform_batch, n_pe_batch


def generator_for_training(
        n_event=None, batch_size=1, n_sample=90, n_sample_init=20,
        pe_rate_mhz=100, bin_size_ns=0.5, sampling_rate_mhz=250,
        amplitude_gain=5., noise_lsb=1.05, sigma_smooth_pe_ns=0.
):
    """
    Generator returning for each iteration a batch of waveforms and a batch of
    pes. The MC simulation is done at each call, no repeating of events
    occurs.
    :param n_event: number of event returned. If None the generator never stops
    to return events.
    :param batch_size: size of the batch size
    :param n_sample: number of sample to simulate. Must be larger than
    n_sample_init. Only n_sample - n_sample_init samples are returned for each
    waveform.
    :param n_sample_init: number of samples to skip at the beginning to take
    into account for effect of photo-electrons before the start of the window.
    :param pe_rate_mhz: rate of pe to simulate. Can be a tuple, then
    for each event of all batches the rate is taken in the range given by the
    tuple.
    :param bin_size_ns: size of bins in nanoseconds for the pes array.
    :param sampling_rate_mhz: sampling rate for the waveforms array.
    :param amplitude_gain: amplitude of a 1.pe. peak in LSB.
    :param noise_lsb: amplitude of random noise to add to the waveforms. Can be a tuple, then
    for each event of all batches the rate is taken in the range given by the
    tuple.
    :param sigma_smooth_pe_ns: width of the gaussian kernel to
    convolve with the batch of pes. Use to convert the position of pe
    to a probability of with sigma_smooth_pe_ns. No convolution is done if
    sigma_smooth_pe_ns <= 0.
    :return: a tuple of 2 arrays at each iteration. First array is a batch of
    waveforms (amplitude of the sampled signal) and a batch of pes (number
    of pes per bin). For both, first dimension is the batch iteration,
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
    gauss_kernel = None
    if sigma_smooth_pe_ns > 0:
        nbin_5sigma = int(np.round(sigma_smooth_pe_ns * 5 / bin_size_ns))
        t_kernel = np.arange(-nbin_5sigma, nbin_5sigma + .1, 1) * bin_size_ns
        gauss_kernel = np.exp(-0.5 * (t_kernel/ sigma_smooth_pe_ns) ** 2)
        gauss_kernel /= np.sum(gauss_kernel)
        print("gaussian kernel prepared, shape=", gauss_kernel.shape)
    if n_event is None:
        while True:
            waveform_batch, n_pe_batch = _get_batch(
                batch_size, n_sample, n_bin_per_sample, pe_rate_mhz,
                bin_size_ns, template_amplitude_bin, noise_lsb, n_sample_init,
                kernel=gauss_kernel
            )
            yield (waveform_batch, n_pe_batch)
    else:
        for event in range(n_event):
            waveform_batch, n_pe_batch = _get_batch(
                batch_size, n_sample, n_bin_per_sample, pe_rate_mhz,
                bin_size_ns, template_amplitude_bin, noise_lsb, n_sample_init,
                kernel=gauss_kernel
            )
            yield (waveform_batch, n_pe_batch)


def plot_example(
        n_event=1, batch_size=1, n_sample=90, n_sample_init=20,
        pe_rate_mhz=100, bin_size_ns=0.5, sampling_rate_mhz=250,
        amplitude_gain=5., noise_lsb=1.05
):
    from matplotlib import pyplot as plt

    generator = generator_for_training(
        n_event=n_event, batch_size=batch_size, n_sample=n_sample,
        n_sample_init=n_sample_init,
        pe_rate_mhz=pe_rate_mhz, bin_size_ns=bin_size_ns,
        sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb
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
    

if __name__ == '__main__':
    plot_example(
        n_event=3, batch_size=2, n_sample=90, n_sample_init=20,
        pe_rate_mhz=20, bin_size_ns=0.5, sampling_rate_mhz=250,
        amplitude_gain=5., noise_lsb=1.05
    )
