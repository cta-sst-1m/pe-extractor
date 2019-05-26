from toy import generator_for_training, n_pe_from_rate, gauss_kernel, model_predict
import numpy as np


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
            pe_pixels = pe_uncor_pixelss[:, :, delay_bin:] + pe_cor[:, :-delay_bin]
        elif delay_sample < 0:
            waveform_pixels = waveform_uncor_pixels[:, :, :delay_sample] + waveform_cor[:, -delay_sample:]
            delay_bin = delay_sample * n_bin_per_sample
            pe_pixels = pe_uncor_pixels[:, :, :delay_bin] + pe_cor[:, -delay_bin:]
        yield waveform_pixels, pe_pixels


def generator_coherent_pixels(
        n_event=None, batch_size=1, n_sample=90, n_sample_init=20, coherant_rate_mhz=10,
        uncoherant_rate_mhz=90, coherant_noise_lsb=0.1, uncoherant_noise_lsb=0.9, n_pixel=2, 
        bin_size_ns=0.5, sampling_rate_mhz=250., amplitude_gain=5.0
):
    generator_coher = generator_for_training(
        n_event=n_event, batch_size=batch_size, n_sample=n_sample,
        n_sample_init=n_sample_init,
        pe_rate_mhz=coherant_rate_mhz, bin_size_ns=bin_size_ns,
        sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=coherant_noise_lsb
    )
    generators_uncoher = []
    for pixel in range(n_pixel):
        generator_uncoher = generator_for_training(
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


def plot_g2(
        run_name=None, shift_in_bins=None, filename=None, batch_size=1, 
        n_sample=2500, bin_size_ns=0.5, sampling_rate_mhz=250., coherant_rate_mhz=100,
        uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0,
        sigma_ns=2.
):
    from matplotlib import pyplot as plt

    if shift_in_bins is None:
        shift_in_bins=np.arange(-50, 50, dtype=int)
    else:
        shift_in_bins = np.array(shift_in_bins, dtype=int)
    sample_length_ns = 1000. / sampling_rate_mhz
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
    g2_pix12_wf, _ = calculate_g2(
        waveform_pixels[0, : :], waveform_pixels[1, : :], shift_in_bins=shift_in_bins
    )
    g2_pix12_proba, _ = calculate_g2(
        proba_pix1, proba_pix2, shift_in_bins=shift_in_bins
    )
    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    axes[0].set_title(title_str)
#    axes[0].plot(shift_in_bins * bin_size_ns, g2_pix11_pe, label='$g^2(pe_1, pe_1)$')
#    axes[0].plot(shift_in_bins * bin_size_ns, g2_pix22_pe, label='$g^2(pe_2, pe_2)$')
    axes[0].plot(shift_in_bins * bin_size_ns, g2_pix12_pe, label='$g^2([pe_1, pe_2)$')
    axes[0].set_xlabel('shift [ns]')
    axes[0].set_ylabel('g2')
    axes[0].legend()

#    axes[1].plot(shift_in_bins * sample_length_ns, g2_pix11_wf, label='$g^2(wf_1, wf_1)$')
#    axes[1].plot(shift_in_bins * sample_length_ns, g2_pix22_wf, label='$g2(wf_2, wf_2)$')
    axes[1].plot(shift_in_bins * sample_length_ns, g2_pix12_wf, label='$g2(wf_1, wf_2)$')
    axes[1].set_xlabel('shift [ns]')
    axes[1].set_ylabel('g2')
    axes[1].legend()

    # axes[2].plot(shift_in_bins * bin_size_ns, g2_pix11_proba, label='$g^2(P_1, P_1)$')
    # axes[2].plot(shift_in_bins * bin_size_ns, g2_pix22_proba, label='$g2(P_2, P_2)$')
    axes[2].plot(shift_in_bins * bin_size_ns, g2_pix12_proba, label='$g2(P_1, P_2)$')
    axes[2].set_xlabel('shift [ns]')
    axes[2].set_ylabel('g2')
    axes[2].legend()
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close(fig)


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
    plot_nr = 1
    plot_g2(run_name, filename=str(plot_nr) + '-co100MHz_unco0_noiseco0_noiseunco0.png', coherant_rate_mhz=100, uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co10MHz_unco0_noiseco0_noiseunco0.png', coherant_rate_mhz=10,uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co1MHz_unco0_noiseco0_noiseunco0.png', coherant_rate_mhz=1,uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co100MHz_unco0_noiseco0_noiseunco0_wf10.png', coherant_rate_mhz=100,uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=10); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co100MHz_unco0_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=0, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co100MHz_unco1MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=1, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co100MHz_unco10MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=10, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co100MHz_unco100MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=100, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co10MHz_unco100MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=10, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co1MHz_unco100MHz_noiseco0_noiseunco0_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=0, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co1MHz_unco100MHz_noiseco0_noiseunco1_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=0., uncoherant_noise_lsb=1, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co1MHz_unco100MHz_noiseco1_noiseunco0_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=1., uncoherant_noise_lsb=0, batch_size=100); plot_nr += 1
    plot_g2(run_name, filename=str(plot_nr) + '-co1MHz_unco100MHz_noiseco0.05_noiseunco1_wf100.png', coherant_rate_mhz=1, uncoherant_rate_mhz=100, coherant_noise_lsb=0.05, uncoherant_noise_lsb=1, batch_size=100); plot_nr += 1


if __name__ == '__main__':
    plots_g2()
    #test_correlated_generator()
