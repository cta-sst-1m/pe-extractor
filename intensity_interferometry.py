from toy import generator_nsb, gauss_kernel, model_predict, read_experimental
import numpy as np
#from matplotlib import use as mpl_use
#mpl_use('Agg')
from matplotlib import pyplot as plt
from train_cnn import loss_all, loss_cumulative, loss_chi2, \
    loss_continuity


def g2_dt0(a1, a2):
    n = np.size(a1)
    return n * np.sum(a1 * a2) / (np.sum(a1) * np.sum(a2))


def calculate_g2(batch_signal1, batch_signal2, shift_in_bins=range(-100, 101)):
    g2 = np.zeros(len(shift_in_bins))
    for b, bin_diff in enumerate(shift_in_bins):
        if bin_diff == 0:
            g2[b] = g2_dt0(batch_signal1, batch_signal2)
        elif bin_diff > 0:
            g2[b] = g2_dt0(batch_signal1[:, bin_diff:], batch_signal2[:, :-bin_diff])
        elif bin_diff < 0:
            g2[b] = g2_dt0(batch_signal1[:, :bin_diff], batch_signal2[:, -bin_diff:])
    return g2, shift_in_bins


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


def plot_g2_exp(rootfile_data, rootfile_hvoff, run_name=None, filename=None,
                shift_in_sample=None, sampling_rate_mhz=250,
                start=None, stop=None, step=None,
                skip_bins=56):
    if shift_in_sample is None:
        shift_in_sample = np.arange(-25, 25, dtype=int)
    else:
        shift_in_sample = np.array(shift_in_sample, dtype=int)
    print('reading HV off data ...')
    wf0_hvoff, wf1_hvoff = read_experimental(
        rootfile_hvoff, start=start, stop=stop, step=step
    )
    baseline_pix0 = np.mean(wf0_hvoff)
    e_noise_pix0 = np.std(wf0_hvoff)
    del wf0_hvoff
    print(
        'pix0: baseline=', str(baseline_pix0),
        ' electronic noise=', str(e_noise_pix0)
    )
    baseline_pix1 = np.mean(wf1_hvoff)
    e_noise_pix1 = np.std(wf1_hvoff)
    del wf1_hvoff
    print(
        'pix1: baseline=', str(baseline_pix1),
        ' electronic noise=', str(e_noise_pix1)
    )
    print('reading data ...')
    wf0, wf1 = read_experimental(
        rootfile_data, start=start, stop=stop, step=step
    )
    n_sample = wf0.shape[1]
    wf0 -= baseline_pix0 - np.mean(wf0[0, :]) + np.tile(np.mean(wf0, axis=1, keepdims=True), [1, n_sample])
    wf1 -= baseline_pix1 - np.mean(wf1[0, :]) + np.tile(np.mean(wf1, axis=1, keepdims=True), [1, n_sample])
    print('calculating g2 on waveforms ...')
    g2_pix12_wf, _ = calculate_g2(wf0, wf1, shift_in_bins=shift_in_sample)
    sample_length_ns = 1000 / sampling_rate_mhz
    shift_waveform_ns = shift_in_sample * sample_length_ns
    title_str = '{} waveforms, {} samples @ {} MHz ({:.2g} s)'.format(
        wf0.shape[0], wf0.shape[1], sampling_rate_mhz,
        wf0.shape[0] * wf0.shape[1] * 1e-9 * sample_length_ns
    )
    title_str += '\nproba from model: {}'.format(run_name)
    g2_pix12_proba = None
    shift_bins_ns = None
    if run_name is not None:
        import tensorflow as tf

        model = tf.keras.models.load_model(
            './Model/' + run_name + '.h5',
            custom_objects={
                'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
                'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
            }
        )
        print('calculating proba with model ' + run_name + ' ...')
        proba0 = model_predict(model, wf0, skip_bins=skip_bins)
        proba1 = model_predict(model, wf1, skip_bins=skip_bins)
        n_bin_per_sample = int(np.round(proba0.shape[-1] / wf0.shape[-1]))
        shift_in_bins = shift_in_sample * n_bin_per_sample
        print('calculating g2 on proba ...')
        g2_pix12_proba, _ = calculate_g2(proba0, proba1,
                                         shift_in_bins=shift_in_bins)
        shift_bins_ns = shift_in_bins * sample_length_ns / n_bin_per_sample
    print('plotting ...')
    fig, axes = plot_calculated_g2(
        shift_wf_ns=shift_waveform_ns, g2_pix12_wf=g2_pix12_wf,
        shift_proba_ns=shift_bins_ns, g2_pix12_proba=g2_pix12_proba,
        title_str=title_str
    )
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        print(filename, 'created')
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
    g2_pix12_pe, _ = calculate_g2(
        pe_pixels[0, : :], pe_pixels[1, : :], shift_in_bins=shift_in_bins
    )
    shift_in_sample_float = shift_in_bins / n_bin_per_sample
    is_integer = np.abs(shift_in_sample_float%1) < 0.01
    shift_in_sample = np.round(shift_in_sample_float[is_integer]).astype(int)
    g2_pix12_wf, _ = calculate_g2(
        waveform_pixels[0, : :], waveform_pixels[1, : :], shift_in_bins=shift_in_sample
    )
    g2_pix12_proba, _ = calculate_g2(
        proba_pix1, proba_pix2, shift_in_bins=shift_in_bins
    )
    shift_bins_ns = shift_in_bins * bin_size_ns
    shift_waveform_ns = shift_in_sample * sample_length_ns
    fig, axes = plot_calculated_g2(
        shift_pe_ns=shift_bins_ns, g2_pix12_pe=g2_pix12_pe,
        shift_wf_ns=shift_waveform_ns, g2_pix12_wf=g2_pix12_wf,
        shift_proba_ns=shift_bins_ns, g2_pix12_proba=g2_pix12_proba,
        title_str=title_str
    )
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        print(filename, 'created')
    plt.close(fig)


def plot_calculated_g2(
        shift_pe_ns=None, g2_pix12_pe=None,
        shift_wf_ns=None, g2_pix12_wf=None,
        shift_proba_ns=None, g2_pix12_proba=None,
        title_str=None
):
    n_row_plot = np.sum([
        shift_pe_ns is not None,
        g2_pix12_wf is not None,
        g2_pix12_proba is not None
    ])
    fig, axes = plt.subplots(
        n_row_plot, 1, figsize=(8, 6), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    if title_str is not None:
        axes[0].set_title(title_str)
    current_axe = 0
    if g2_pix12_pe is not None:
        axes[current_axe].plot(shift_pe_ns, g2_pix12_pe - 1,
                               label='$g^2([pe_1, pe_2)$')
        axes[current_axe].set_ylabel('g2 - 1 ')
        axes[current_axe].legend()
        current_axe += 1
    if g2_pix12_wf is not None:
        axes[current_axe].plot(shift_wf_ns, g2_pix12_wf - 1,
                               label='$g2(wf_1, wf_2)$')
        axes[current_axe].set_xlabel('shift [ns]')
        axes[current_axe].set_ylabel('g2 - 1 ')
        axes[current_axe].legend()
        current_axe += 1
    if g2_pix12_proba is not None:
        axes[current_axe].plot(shift_proba_ns, g2_pix12_proba - 1,
                               label='$g2(P_1, P_2)$')
        axes[current_axe].set_xlabel('shift [ns]')
        axes[current_axe].set_ylabel('g2 - 1')
        axes[current_axe].legend()
        current_axe += 1
    axes[current_axe-1].set_xlabel('shift [ns]')
    plt.tight_layout()
    return fig, axes


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


if __name__ == '__main__':
    # import tensorflow as tf

    # tf.enable_eager_execution()
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
    n_events = 1
    files_hvoff = 'experimental_waveforms/SST1M_01_20190523_0000_0000_raw_waveforms.root'
    files_pp = 'experimental_waveforms/SST1M_01_20190523_0004_0004_raw_waveforms.root'
    plot_g2_exp(
        files_pp, files_hvoff,
        run_name='cnn-example', filename='exp_pp_' + str(n_events) + '.png',
        shift_in_sample=np.arange(-200, 200),
        start=None, stop=n_events, step=None, skip_bins=56
    )
    files_sp = 'experimental_waveforms/SST1M_01_20190523_0104_0108_raw_waveforms.root'
    plot_g2_exp(
        files_sp, files_hvoff,
        run_name='cnn-example', filename='exp_sp_' + str(n_events) + '.png',
        shift_in_sample=np.arange(-200, 200),
        start=None, stop=n_events, step=None, skip_bins=56
    )
    #plots_g2('cnn-example')
    #test_correlated_generator()
