import sys
import gc
from pe_extractor.toy import generator_nsb, gauss_kernel, model_predict, read_experimental
import numpy as np
#from matplotlib import use as mpl_use
#mpl_use('Agg')
from matplotlib import pyplot as plt
from pe_extractor.train_cnn import loss_all, loss_cumulative, loss_chi2, \
    loss_continuity
from astropy.stats import sigma_clip

def _getr(slist, olist, seen):
  for e in slist:
    if id(e) in seen:
      continue
    seen[id(e)] = None
    olist.append(e)
    tl = gc.get_referents(e)
    if tl:
      _getr(tl, olist, seen)


# The public function.
def get_all_objects():
  """Return a list of all live Python
  objects, not including the list itself."""
  gcl = gc.get_objects()
  olist = []
  seen = {}
  # Just in case:
  seen[id(gcl)] = None
  seen[id(olist)] = None
  seen[id(seen)] = None
  # _getr does the real work.
  _getr(gcl, olist, seen)
  return olist


def memory_debug(variables, str):
    print(str, ':')
    if isinstance(variables, list):
        for var in variables:
            size = sys.getsizeof(var)
            if size > 1e6:
                print(size * 1e-6, 'Mb')
    elif isinstance(variables, dict):
        names = list(variables.keys())
        for name in names:
            size = sys.getsizeof(variables[name])
            if size > 1e6:
                print(name, size * 1e-6, 'Mb')


def calculate_correlation(
        batch_signal1, batch_signal2, shift_in_bins=range(-100, 101)
):
    sum1 = np.zeros(len(shift_in_bins))
    sum2 = np.zeros(len(shift_in_bins))
    sum12 = np.zeros(len(shift_in_bins))
    sum11 = np.zeros(len(shift_in_bins))
    sum22 = np.zeros(len(shift_in_bins))
    n_sample = np.zeros(len(shift_in_bins), dtype=int)
    n_event, n_sample_batch = batch_signal1.shape
    if np.max(np.abs(shift_in_bins)) >= n_sample_batch:
        raise ValueError(
            "the shift in bin given is too large " +
            "for the current number of samples."
        )
    for b, bin_diff in enumerate(shift_in_bins):
        if bin_diff == 0:
            signal1_shifted = batch_signal1
            signal2_shifted = batch_signal2
        elif bin_diff > 0:
            signal1_shifted = batch_signal1[:, bin_diff:]
            signal2_shifted = batch_signal2[:, :-bin_diff]
        else:  # bin_diff < 0
            signal1_shifted = batch_signal1[:, :bin_diff]
            signal2_shifted = batch_signal2[:, -bin_diff:]
        mask = np.logical_and(
            np.isfinite(signal1_shifted),
            np.isfinite(signal2_shifted)
        )
        n_sample[b] = np.sum(mask)
        s1_finite = signal1_shifted[mask]
        s2_finite = signal2_shifted[mask]
        sum1[b] = np.sum(s1_finite)
        sum2[b] = np.sum(s2_finite)
        sum12[b] = np.sum(s1_finite * s2_finite)
        sum11[b] = np.sum(s1_finite ** 2)
        sum22[b] = np.sum(s2_finite ** 2)
    return sum1, sum2, sum12, sum11, sum22, n_sample


def calculate_g2(batch_signal1, batch_signal2, shift_in_bins):
    sum1, sum2, sum12, sum11, sum22, n_sample = calculate_correlation(
        batch_signal1, batch_signal2, shift_in_bins=shift_in_bins
    )
    g2 = n_sample * sum12 / (sum1 * sum2)
    return g2


def correlate_pixels(generator_cor, *generator_uncor_pixels, delay_sample=0, n_bin_per_sample=8):
    n_pixel = len(generator_uncor_pixels)
    if n_pixel < 2:
        raise ValueError('generator_correlated_pixels takes at least 3 arguments')
    for batch_cor, *batch_uncor in zip(generator_cor, *generator_uncor_pixels):
        waveform_cor, pe_cor = batch_cor
        # print('waveform_cor:', waveform_cor.shape)
        # print('pe_cor:', pe_cor.shape)
        waveform_uncor_pixels = np.stack([waveform for waveform, _ in batch_uncor])
        pe_uncor_pixels = np.stack([pe for _, pe in batch_uncor])
        # print('waveform_uncor_pixels:',  waveform_uncor_pixels.shape)
        if delay_sample == 0:
            waveform_pixels = waveform_uncor_pixels + waveform_cor
            pe_pixels = pe_uncor_pixels + pe_cor
        elif delay_sample > 0:
            waveform_uncor_shifted = waveform_uncor_pixels[:, :, delay_sample:]
            waveform_cor_shifted = waveform_cor[:, :-delay_sample]
            waveform_pixels = waveform_uncor_shifted + waveform_cor_shifted
            delay_bin = delay_sample * n_bin_per_sample
            pe_pixels = pe_uncor_pixels[:, :, delay_bin:] + pe_cor[:, :-delay_bin]
        else:
            waveform_pixels = waveform_uncor_pixels[:, :, :delay_sample] + waveform_cor[:, -delay_sample:]
            delay_bin = delay_sample * n_bin_per_sample
            pe_pixels = pe_uncor_pixels[:, :, :delay_bin] + pe_cor[:, -delay_bin:]
        yield waveform_pixels, pe_pixels


def generator_coherent_pixels(
        n_event=None, batch_size=1, n_sample=90, n_sample_init=20, coherant_rate_mhz=10,
        uncoherant_rate_mhz=90, coherant_noise_lsb=0.1, uncoherant_noise_lsb=0.9, n_pixel=2, 
        bin_size_ns=0.5, sampling_rate_mhz=250., amplitude_gain=5.0
):
    generator_coher = generator_nsb(
        n_event=n_event, batch_size=batch_size, n_sample=n_sample,
        n_sample_init=n_sample_init,
        pe_rate_mhz=coherant_rate_mhz, bin_size_ns=bin_size_ns,
        sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=coherant_noise_lsb
    )
    generators_uncoher = []
    for pixel in range(n_pixel):
        generator_uncoher = generator_nsb(
            n_event=n_event, batch_size=batch_size, n_sample=n_sample,
            n_sample_init=n_sample_init,
            pe_rate_mhz=uncoherant_rate_mhz, bin_size_ns=bin_size_ns,
            sampling_rate_mhz=sampling_rate_mhz,
            amplitude_gain=amplitude_gain, noise_lsb=uncoherant_noise_lsb
        )
        generators_uncoher.append(generator_uncoher)
    generator_pixels = correlate_pixels(
        generator_coher, *generators_uncoher,
        delay_sample=0
    )
    return generator_pixels


def plot_g2_exp(rootfile_data, rootfile_hvoff, run_name=None, g2_plot=None,
                shift_in_sample=None, sampling_rate_mhz=250,
                start=0, stop=None, step=None, skip_bins=64,
                n_waveform_max=100, g2_file=None, n_bin_per_sample=8
):
    import ROOT
    sample_length_ns = 1000. / sampling_rate_mhz
    bin_length_ns = sample_length_ns / n_bin_per_sample
    f = ROOT.TFile(rootfile_data)
    n_waveform_files = f.waveforms.GetEntries()
    f.Close()
    if shift_in_sample is None:
        shift_in_sample = np.arange(-25, 25, dtype=int)
    else:
        shift_in_sample = np.array(shift_in_sample, dtype=int)
    if stop is None or stop > n_waveform_files:
        stop = n_waveform_files
    print('reading HV off data ...')
    wf0_hvoff, wf1_hvoff, _, _= read_experimental(
        rootfile_hvoff, start=0, stop=1000, step=step
    )
    baseline_pix0 = np.nanmean(
        get_baseline(wf0_hvoff, margin_lsb=5, samples_around=4)
    )
    e_noise_pix0 = np.std(wf0_hvoff)
    print(
        'pix0: baseline=', str(baseline_pix0),
        ' electronic noise=', str(e_noise_pix0)
    )
    baseline_pix1 = np.nanmean(
        get_baseline(wf1_hvoff, margin_lsb=5, samples_around=4)
    )
    e_noise_pix1 = np.std(wf1_hvoff)
    print(
        'pix1: baseline=', str(baseline_pix1),
        ' electronic noise=', str(e_noise_pix1)
    )
    del wf0_hvoff
    del wf1_hvoff
    print('reading data (', stop-start, '/', n_waveform_files, 'waveforms)...')
    sum1_wf = np.zeros_like(shift_in_sample, dtype=float)
    sum2_wf = np.zeros_like(shift_in_sample, dtype=float)
    sum12_wf = np.zeros_like(shift_in_sample, dtype=float)
    sum11_wf = np.zeros_like(shift_in_sample, dtype=float)
    sum22_wf = np.zeros_like(shift_in_sample, dtype=float)
    n_sample_wf = np.zeros_like(shift_in_sample, dtype=int)
    if start:
        current_start = start
    else:
        current_start = 0
    model = None
    sum1_pb = None
    sum2_pb = None
    sum12_pb = None
    sum11_pb = None
    sum22_pb = None
    n_sample_pb = None
    if run_name is not None:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            './Model/' + run_name + '.h5',
            custom_objects={
                'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
                'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
            }
        )
        shift_in_bins = shift_in_sample[:, None] * n_bin_per_sample + np.arange(-4, 4)[None, :]
        shift_in_bins = shift_in_bins.flatten()
        shift_bin_ns = shift_in_bins * bin_length_ns

        sum1_pb = np.zeros_like(shift_in_bins, dtype=float)
        sum2_pb = np.zeros_like(shift_in_bins, dtype=float)
        sum12_pb = np.zeros_like(shift_in_bins,dtype=float)
        sum11_pb = np.zeros_like(shift_in_bins, dtype=float)
        sum22_pb = np.zeros_like(shift_in_bins, dtype=float)
        n_sample_pb = np.zeros_like(shift_in_bins, dtype=int)
    n_waveform_tot = 0
    shift_waveform_ns = shift_in_sample * sample_length_ns
    while True:
        current_stop = current_start + n_waveform_max
        if stop is not None and current_stop > stop:
            current_stop = stop
        if current_stop == current_start:
            break
        wf0, wf1, t, _ = read_experimental(
            rootfile_data, start=current_start, stop=current_stop, step=step
        )
        n_waveform, n_sample = wf0.shape
        if n_waveform == 0:
            break
        n_waveform_tot += n_waveform
        wf0 = wf0 - baseline_pix0
        wf1 = wf1 - baseline_pix1
        print('calculating g2 on waveforms', current_start, 'to',
              current_stop - 1 , 'with', n_sample, 'samples...')
        sum1, sum2, sum12, sum11, sum22, n_samp = calculate_correlation(
            wf0,
            wf1,
            shift_in_bins=shift_in_sample
        )
        sum1_wf += sum1
        sum2_wf += sum2
        sum12_wf += sum12
        sum11_wf += sum11
        sum22_wf += sum22
        n_sample_wf += n_samp
        if model is not None:
            # print('calculating proba for 1st pixel with model', run_name)
            proba0 = model_predict(model, wf0, skip_bins=skip_bins)
            # print('calculating proba for 2nd pixel with model', run_name)
            proba1 = model_predict(model, wf1, skip_bins=skip_bins)
            # print('calculating g2 on proba ...')
            sum1, sum2, sum12, sum11, sum22, n_samp = calculate_correlation(
                proba0,
                proba1,
                shift_in_bins=shift_in_bins
            )
            sum1_pb += sum1
            sum2_pb += sum2
            sum12_pb += sum12
            sum11_pb += sum11
            sum22_pb += sum22
            n_sample_pb += n_samp
        current_start = current_stop  # prepare for next iteration

    if model is not None:
        del model
    if g2_file is not None:
        np.savez(
            g2_file, shift_in_sample=shift_in_sample,
            n_sample_wf=n_sample_wf, sum1_wf=sum1_wf, sum2_wf=sum2_wf,
            sum12_wf=sum12_wf, sum11_wf=sum11_wf, sum22_wf=sum22_wf,
            shift_in_bins=shift_in_bins,
            n_sample_pb=n_sample_pb, sum1_pb=sum1_pb, sum2_pb=sum2_pb,
            sum12_pb=sum12_pb, sum11_pb=sum11_pb, sum22_pb=sum22_pb,
            baseline_pix0=baseline_pix0, baseline_pix1=baseline_pix1,
            e_noise_pix0=e_noise_pix0, e_noise_pix1=e_noise_pix1
        )
    # g2_pix12_wf = n_sample_wf * sum12_wf / (sum1_wf * sum2_wf)
    # g2_pix11_wf = n_sample_wf * sum11_wf / (sum1_wf ** 2)
    # g2_pix22_wf = n_sample_wf * sum22_wf / (sum2_wf ** 2)

    g2_wf = n_sample_wf * sum12_wf / (sum1_wf * sum2_wf)
    title_str = '{} waveforms, {} samples @ {} MHz ({:.2g} s)'.format(
        n_waveform_tot, n_sample, sampling_rate_mhz,
        n_waveform_tot * n_sample * 1e-9 * sample_length_ns
    )
    g2_proba = None
    if run_name is not None:
        #g2_proba = n_sample_pb * (sum11_pb + sum22_pb + 2 * sum12_pb) / ((sum1_pb + sum2_pb) ** 2)
        g2_proba = n_sample_pb * sum12_pb / (sum1_pb * sum2_pb)
        title_str += '\nproba from model: {}'.format(run_name[:30])

    print('plotting ...')

    fig, axes = plot_calculated_g2(
        shift_pe_ns=None, g2_pix12_pe=None,
        shift_wf_ns=shift_waveform_ns, g2_pix12_wf=g2_wf,
        shift_proba_ns=shift_bin_ns, g2_pix12_proba=g2_proba,
        title_str=title_str
    )
    if g2_plot is None:
        plt.show()
    else:
        plt.savefig(g2_plot)
        print(g2_plot, 'created')
    plt.close(fig)


def plot_g2_toy(
        run_name=None, shift_in_bins=None, filename=None, batch_size=1, 
        n_sample=2500, bin_size_ns=0.5, sampling_rate_mhz=250., coherant_rate_mhz=100,
        uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0,
        sigma_ns=2.
):
    if shift_in_bins is None:
        shift_in_bins=np.arange(-200, 200, dtype=int)
    else:
        shift_in_bins = np.array(shift_in_bins, dtype=int)
    sample_length_ns = 1000. / sampling_rate_mhz
    n_bin_per_sample = sample_length_ns / bin_size_ns
    title_str = '{} waveforms, {} samples @ {} MHz ({:.2g} s)'.format(
        batch_size, n_sample, sampling_rate_mhz, 
        batch_size * n_sample * 1e-9 * sample_length_ns
    )
    title_str += '\n{} MHz of coherent p.e., {} MHz of uncoherent p.e.'.format(
        coherant_rate_mhz, uncoherant_rate_mhz
    )
    title_str += '\n{} LSB of coherent noise, {} LSB of uncoherent noise'.format(
        coherant_noise_lsb, uncoherant_noise_lsb
    )

    generator_pixels = generator_coherent_pixels(
        n_event=1, batch_size=batch_size, n_sample=n_sample, n_sample_init=20,
        coherant_rate_mhz=coherant_rate_mhz, uncoherant_rate_mhz=uncoherant_rate_mhz,
        coherant_noise_lsb=coherant_noise_lsb,
        uncoherant_noise_lsb=uncoherant_noise_lsb, n_pixel=2, bin_size_ns=bin_size_ns,
        sampling_rate_mhz=sampling_rate_mhz, amplitude_gain=5.0
    )
    waveform_pixels, pe_pixels = next(generator_pixels)
    del generator_pixels
    if run_name is not None:
        import tensorflow as tf

        model = tf.keras.models.load_model(
            './Model/' + run_name + '.h5',
            custom_objects={
                'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
                'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
            }
        )
        proba_pix1 = model_predict(model, waveform_pixels[0, : :])
        proba_pix2 = model_predict(model, waveform_pixels[1, : :])
        title_str += '\nproba from model: {}'.format(run_name)
        del model
    else:
        title_str += '\nproba with $\sigma={}$ ns'.format(sigma_ns)
        kernel = gauss_kernel(bin_size_ns=bin_size_ns, sigma_ns=sigma_ns)
        proba_pix1 = np.zeros_like(pe_pixels[0, :, :])
        proba_pix2 = np.zeros_like(pe_pixels[1, :, :])
        for b in range(batch_size):
            proba_pix1[b, :] = np.convolve(pe_pixels[0, b, :], kernel, 'same')
            proba_pix2[b, :] = np.convolve(pe_pixels[1, b, :], kernel, 'same')
    g2_pix12_pe = calculate_g2(
        pe_pixels[0, : :], pe_pixels[1, : :], shift_in_bins=shift_in_bins
    )
    shift_in_sample_float = shift_in_bins / n_bin_per_sample
    is_integer = np.abs(shift_in_sample_float%1) < 0.01
    shift_in_sample = np.round(shift_in_sample_float[is_integer]).astype(int)
    g2_pix12_wf = calculate_g2(
        waveform_pixels[0, : :], waveform_pixels[1, : :], shift_in_bins=shift_in_sample
    )
    g2_pix12_proba = calculate_g2(
        proba_pix1, proba_pix2, shift_in_bins=shift_in_bins
    )
    shift_samples_ns = shift_in_bins * bin_size_ns
    shift_waveform_ns = shift_in_sample * sample_length_ns
    fig, axes = plot_calculated_g2(
        shift_pe_ns=shift_samples_ns, g2_pix12_pe=g2_pix12_pe,
        shift_wf_ns=shift_samples_ns, g2_pix12_wf=g2_pix12_wf,
        shift_proba_ns=shift_samples_ns, g2_pix12_proba=g2_pix12_proba,
        title_str=title_str
    )
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        print(filename, 'created')
    plt.close(fig)


def get_stat_g2(g2):
    filtered = sigma_clip(g2, sigma=3, maxiters=10)
    mask_in_baseline = ~filtered.mask
    baseline_data = g2[mask_in_baseline]
    mean_g2 = np.mean(baseline_data)
    std_g2 = np.std(baseline_data)
    if np.all(mask_in_baseline):
        peak_max = 0
        peak_pos = 0
    else:
        peak_data = np.zeros_like(g2)
        peak_data[~mask_in_baseline] = g2[~mask_in_baseline]
        peak_pos = np.argmax(peak_data)
        peak_max = peak_data[peak_pos]
    return peak_max, mean_g2, std_g2, mask_in_baseline, peak_pos


def plot_calculated_g2(
        shift_pe_ns=None, g2_pix12_pe=None,
        shift_wf_ns=None, g2_pix12_wf=None,
        shift_proba_ns=None, g2_pix12_proba=None,
        title_str=None, ax=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()
    if title_str is not None:
        ax.set_title(title_str)
    if g2_pix12_pe is not None:
        peak_max_pe, mean_g2_pe, std_g2_pe, mask_pe, peak_pos_pe = get_stat_g2(
            g2_pix12_pe)
        g2_ampl_pe = peak_max_pe - mean_g2_pe
        label_pe = '$g^2_{pe_1, pe_2} , g^2(\delta t=' + \
                   '{}'.format(shift_pe_ns[peak_pos_pe]) + ')=' + \
                   '{:.1e}'.format(g2_ampl_pe) + ' \pm ' + \
                   '{:.0e}'.format(std_g2_pe) + '$'
        ax.plot(
            shift_pe_ns, g2_pix12_pe - mean_g2_pe, 'g-',label=label_pe
        )
        ax.plot(
            shift_pe_ns[mask_pe], g2_pix12_pe[mask_pe] - mean_g2_pe, 'g.',
            label=None
        )
        ax.plot(
            shift_pe_ns[~mask_pe], g2_pix12_pe[~mask_pe] - mean_g2_pe, 'g+',
            label=None
        )
    if g2_pix12_proba is not None:
        peak_max_pb, mean_g2_pb, std_g2_pb, mask_pb, peak_pos_pb = get_stat_g2(
            g2_pix12_proba)
        g2_ampl_pb = peak_max_pb - mean_g2_pb
        label_pb = '$g^2_{P_1, P_2} , g^2(\delta t=' + \
                   '{}'.format(shift_proba_ns[peak_pos_pb]) + ')=' + \
                   '{:.3e}'.format(g2_ampl_pb) + ' \pm ' + \
                   '{:.3e}'.format(std_g2_pb) + '$'
        ax.plot(
            shift_proba_ns, g2_pix12_proba - mean_g2_pb, 'r-', label=label_pb)
        ax.plot(
            shift_proba_ns[mask_pb],
            g2_pix12_proba[mask_pb] - mean_g2_pb,
            'r.',
            label=None
        )
        ax.plot(
            shift_proba_ns[~mask_pb],
            g2_pix12_proba[~mask_pb] - mean_g2_pb,
            'r+',
            label=None
        )
    if g2_pix12_wf is not None:
        peak_max_wf, mean_g2_wf, std_g2_wf, mask_wf, peak_pos_wf = get_stat_g2(
            g2_pix12_wf)
        g2_ampl_wf = peak_max_wf - mean_g2_wf
        label_wf = '$g^2_{wf_1, wf_2} , g^2(\delta t=' + \
                   '{}'.format(shift_wf_ns[peak_pos_wf]) + ')=' + \
                   '{:.1e}'.format(g2_ampl_wf) + ' \pm ' + \
                   '{:.0e}'.format(std_g2_wf) + '$'
        ax.plot(
            shift_wf_ns, g2_pix12_wf - mean_g2_wf, 'b-',label=label_wf
        )
        ax.plot(
            shift_wf_ns[mask_wf], g2_pix12_wf[mask_wf] - mean_g2_wf, 'b.',
            label=None
        )
        ax.plot(
            shift_wf_ns[~mask_wf], g2_pix12_wf[~mask_wf] - mean_g2_wf, 'b+',
            label=None
        )

    ax.grid(True)
    ax.set_xlabel('shift [ns]')
    ax.set_ylabel('$g^2 - <g^2_{baseline}>$')
    ax.legend()
    plt.tight_layout()
    return fig, ax


def test_correlated_generator(n_pixel=2):
    from matplotlib import pyplot as plt
    batch_size = 1
    bin_size_ns=0.5
    sampling_rate_mhz=250.
    generator_pixels = generator_coherent_pixels(
        n_event=1, batch_size=batch_size, n_sample=900, n_sample_init=20, coherant_rate_mhz=10,
        uncoherant_rate_mhz=1, coherant_noise_lsb=0.1, uncoherant_noise_lsb=0.9, n_pixel=2, 
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz, amplitude_gain=5.0
    )
    fig, axes = plt.subplots(n_pixel, 2, figsize=(8, 6))
    for event, (waveform_pixels, pe_pixels) in enumerate(generator_pixels):
        for pixel in range(n_pixel):
            for b in range(batch_size):
                t_max_ns = pe_pixels.shape[-1] * bin_size_ns
                t_bin = np.arange(0, t_max_ns, bin_size_ns)
                t_waveform = np.arange(0, t_max_ns, 1000 / sampling_rate_mhz)
                axes[pixel, 0].plot(t_bin, pe_pixels[pixel, b, :])
                axes[pixel, 1].plot(t_waveform, waveform_pixels[pixel, b, :])
    plt.show()
    plt.close(fig)


def plots_g2(run_name=None):
    #plot_g2_toy(run_name, filename='1-co1MHz_unco0_noiseco0_noiseunco0.png', coherant_rate_mhz=1,uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0)
    #plot_g2_toy(run_name, filename='2-co10MHz_unco0_noiseco0_noiseunco0.png', coherant_rate_mhz=10,uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0)
    #plot_g2_toy(run_name, filename='3-co100MHz_unco0_noiseco0_noiseunco0.png', coherant_rate_mhz=100, uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0)
    #plot_g2_toy(run_name, filename='4-co100MHz_unco0_noiseco0_noiseunco0_wf10.png', coherant_rate_mhz=100,uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=10)
    #plot_g2_toy(run_name, filename='5-co100MHz_unco0_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100)
    #plot_g2_toy(run_name, filename='6-co100MHz_unco1MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=1, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100)
    #plot_g2_toy(run_name, filename='7-co100MHz_unco10MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=10, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100)
    plot_g2_toy(run_name, filename='8-co100MHz_unco100MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100)
    plot_g2_toy(run_name, filename='9-co10MHz_unco100MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=10, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100)
    plot_g2_toy(run_name, filename='10-co1MHz_unco100MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100)
    plot_g2_toy(run_name, filename='11-co1MHz_unco100MHz_noiseco0_noiseunco1_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=1, batch_size=100)
    plot_g2_toy(run_name, filename='12-co1MHz_unco100MHz_noiseco1_noiseunco0_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=1., uncoherant_noise_lsb=0, batch_size=100)
    plot_g2_toy(run_name, filename='13-co1MHz_unco100MHz_noiseco0.05_noiseunco1_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=0.05, uncoherant_noise_lsb=1, batch_size=100)


def get_baseline(waveforms, margin_lsb=8, samples_around=4):
    min_wf = np.min(waveforms, axis=1, keepdims=True)
    samples_ignored = waveforms > min_wf + margin_lsb
    for k in range(-samples_around, samples_around+1):
        samples_ignored = np.logical_or(
            samples_ignored,
            np.roll(samples_ignored, k, axis=1)
        )
    filtered = np.nan * np.ones_like(waveforms)
    filtered[~samples_ignored] = waveforms[~samples_ignored]
    baselines = np.nanmean(filtered, axis=1)
    return baselines


def get_baseline_points(waveforms, margin_lsb=8, samples_around=4,
                        sampling_rate_mhz=250):
    min_wf = np.min(waveforms, axis=1, keepdims=True)
    samples_ignored = waveforms > min_wf + margin_lsb
    for k in range(-samples_around, samples_around + 1):
        samples_ignored = np.logical_or(
            samples_ignored,
            np.roll(samples_ignored, k, axis=1)
        )
    baselines = waveforms[~samples_ignored]
    t_us = np.arange(waveforms.size) / sampling_rate_mhz
    t_us_baselines = t_us.reshape(waveforms.shape)[~samples_ignored]
    return t_us_baselines, baselines


def plot_baseline(rootfile_data, sampling_rate_mhz=250, n_event_max=None,
                  margin_lsb=10, samples_around=5):
    all_baseline_wf0 = []
    all_baseline_wf1 = []
    t_baseline = []
    start = 0
    max_batch = 10000
    while True:
        if n_event_max:
            stop = n_event_max
        if stop - start > max_batch:
            stop = start + max_batch
        wf0, wf1, t, _ = read_experimental(
            rootfile_data, start=start, stop=stop
        )
        baseline_wf0 = get_baseline(wf0, margin_lsb=margin_lsb,
                                    samples_around=samples_around)
        baseline_wf1 = get_baseline(wf1, margin_lsb=margin_lsb,
                                    samples_around=samples_around)
        all_baseline_wf0.extend(baseline_wf0)
        all_baseline_wf1.extend(baseline_wf1)
        t_baseline.extend(t)
        start += max_batch
        if n_event_max and start >= n_event_max:
            break
        if n_event_max is None and stop - start < max_batch:
            break
    fig = plt.figure(figsize=(8, 6))
    t_baseline = np.arange(len(all_baseline_wf0)) * 1e-6 / sampling_rate_mhz * wf0.shape[1]
    plt.plot(t_baseline, all_baseline_wf0)
    plt.plot(t_baseline, all_baseline_wf1)
    plt.xlabel('t [s]')
    plt.ylabel('baseline')
    plt.savefig('plots/baselines.png')
    print('plots/baselines.png created')
    plt.close(fig)


def plot_baseline_points(rootfile_data, sampling_rate_mhz=250, n_event_max=None):
    wf0, wf1, t, _ = read_experimental(
        rootfile_data, start=0, stop=n_event_max
    )
    t0, baseline_wf0 = get_baseline_points(wf0, margin_lsb=8, samples_around=4,
                                           sampling_rate_mhz=sampling_rate_mhz)
    t1, baseline_wf1 = get_baseline_points(wf1, margin_lsb=8, samples_around=4,
                                           sampling_rate_mhz=sampling_rate_mhz)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(t0, baseline_wf0, '-')
    plt.plot(t1, baseline_wf1, '-')
    plt.xlabel('t [us]')
    plt.ylabel('baseline')
    plt.xlim([0, 3])
    plt.savefig('plots/baselines.png')
    print('plots/baselines.png created')
    plt.close(fig)


if __name__ == '__main__':

    # files_20190520 = [
    #     'experimental_waveforms/SST1M_01_20190520_0010_0014.root',
    #     'experimental_waveforms/SST1M_01_20190520_0015_0019.root',
    #     'experimental_waveforms/SST1M_01_20190520_0050_0054.root',
    #     'experimental_waveforms/SST1M_01_20190520_0055_0059.root',
    # ]
    # plot_g2_exp(
    #     files_20190520,
    #     run_name='cnn-example', filename='exp_20190520_0010-0014.png',
    #     shift_in_sample=np.arange(-50, 50)
    # )
    n_events = 5000
    # June 2019 data
    # files_pp = 'experimental_waveforms/SST1M01_20190625_0000_0000_raw_waveforms.root'
    # files_sp = 'experimental_waveforms/SST1M01_20190625_0013_0013_raw_waveforms.root'
    #files_lampoff = 'experimental_waveforms/SST1M01_20190625_0026_0026_raw_waveforms.root'

    # May 2019 data
    #files_hvoff = 'experimental_waveforms/SST1M_01_20190523_0000_0000_raw_waveforms.root'
    #files_pp = 'experimental_waveforms/SST1M_01_20190523_0004_0008_raw_waveforms.root'
    #files_sp = 'experimental_waveforms/SST1M_01_20190523_0104_0108_raw_waveforms.root'

    # September 17 data
    files_lampoff = 'experimental_waveforms/SST1M_01_20190917_0007_0007_raw_waveforms.root'
    files_pp = 'experimental_waveforms/SST1M_01_20190917_0295_0295_raw_waveforms.root'
    files_sp = 'experimental_waveforms/SST1M_01_20190917_0194_0194_raw_waveforms.root'

    #log_shift = np.unique(np.logspace(3, 7, 50, dtype=int))
    #shift_in_sample = np.sort(np.hstack([-log_shift, 0, log_shift]))
    #shift_in_sample = np.linspace(-5e6, 5e6, 200, dtype=int)
    shift_in_sample = range(-125, 126)

    # plot_baseline(files_pp, n_event_max=n_events, margin_lsb=10, samples_around=2)
    # model = 'deconv_filters-16x20-8x10-4x10-2x10-1x1-1x1-1x1_lr0.0003_rel_gain_std0.1_pos_rate0-200_smooth2.0_noise0-2_baseline0_run0rr'
    # model = 'cnn-example'
    model = 'C16x16_U2_C32x16_U2_C64x8_U2_C128x8_C64x4_C32x4_C16x2_C4x2_C1x1_C1x1_ns0.1_shift64_all1-50-10lr0.0002smooth1_amsgrad_run0'

    if model is None:
        g2_plot = 'plots/g2_off_' + str(n_events) + '.png'
    else:
        g2_plot = 'plots/' + model + '/g2_off_' + str(n_events) + '.png'
    plot_g2_exp(
        files_lampoff, files_lampoff, run_name=model,
        g2_plot=g2_plot, shift_in_sample=shift_in_sample,
        start=0, stop=n_events, step=None, skip_bins=64,
        g2_file=g2_plot.replace('.png', '.npz')
    )

    if model is None:
        g2_plot='plots/g2_pp_' + str(n_events) + '.png'
    else:
        g2_plot='plots/' + model + '/g2_pp_' + str(n_events) + '.png'
    plot_g2_exp(
        files_pp, files_lampoff, run_name=model,
        g2_plot=g2_plot, shift_in_sample=shift_in_sample,
        start=0, stop=n_events, step=None, skip_bins=64,
        g2_file=g2_plot.replace('.png', '.npz'),
    )

    if model is None:
        g2_plot='plots/g2_sp_' + str(n_events) + '.png'
    else:
        g2_plot='plots/' + model + '/g2_sp_' + str(n_events) + '.png'
    plot_g2_exp(
        files_sp, files_lampoff,
        run_name=model,
        g2_plot=g2_plot,
        shift_in_sample=shift_in_sample,
        start=0, stop=n_events, step=None, skip_bins=64,
        g2_file=g2_plot.replace('.png', '.npz'),
    )

    #plots_g2('cnn-example')
    #test_correlated_generator()
