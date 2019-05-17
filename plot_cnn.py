from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from toy import generator_for_training
from train_cnn import loss_all, loss_cumulative, loss_chi2, loss_continuity, \
    timebin_from_prediction
import tensorflow as tf


def sum_norm_gaussian(x_val, offset, *args):
    if len(args) % 2 != 0:
        raise ValueError('size of args must be even.')
    n_gauss = int(len(args) / 2)
    center = args[:n_gauss]
    width = args[n_gauss:]
    n_val = np.size(x_val)
    gauss_cen = np.tile(np.array(center).reshape([-1, 1]), [1, n_val])
    gauss_width = np.tile(np.array(width).reshape([-1, 1]), [1, n_val])
    gauss_norm = np.sqrt(2 * np.pi) * gauss_width
    gauss_x = np.tile(
        np.array(x_val).reshape([1, -1]),
        [n_gauss, 1]
    )
    amp_gaussians = 1 / gauss_norm * np.exp(
        - 0.5 * ((gauss_x - gauss_cen) / gauss_width) ** 2
    )
    return np.sum(amp_gaussians, axis=0) + offset


def fit_sum_norm_gaussian(
        x_array, y_array_gauss, n_gaussian, cen_init=None,
):
    x_min = x_array[0] - 10.
    x_max = x_array[-1] + 10.
    if cen_init is None:
        # distribute Gaussian initial position
        gauss_cen_init = np.linspace(x_array[0], x_array[-1], n_gaussian + 2)[1:-1]
    else:
        cen_init = np.array(cen_init)
        gauss_cen_init = cen_init[cen_init >= x_min]
        gauss_cen_init = gauss_cen_init[gauss_cen_init <= x_max]
        gauss_cen_init = gauss_cen_init[:n_gaussian].tolist()
        n_missing = n_gaussian - len(gauss_cen_init)
        if n_missing > 0:
            print('WARNING:', n_missing, 'missing initial time')
            gauss_cen_init.extend(
                np.linspace(x_array[0], x_array[-1], n_missing + 2)[1:-1]
            )
    bounds = ([-.01], [.01])
    bounds[0].extend(x_min * np.ones(n_gaussian))
    bounds[0].extend(0. * np.ones(n_gaussian))
    bounds[1].extend(x_max * np.ones(n_gaussian))
    bounds[1].extend(10. * np.ones(n_gaussian))
    p0 = [0, ]  # null offset for start of fit
    p0.extend(gauss_cen_init)
    p0.extend(4. * np.ones([n_gaussian]))
    sigma_array_gauss = 1/(.1 + (np.abs(y_array_gauss) >= 0.01).astype(float))
    popt_gauss, pcov_gauss = curve_fit(
        sum_norm_gaussian, x_array, y_array_gauss,
        p0=p0, sigma=sigma_array_gauss, bounds=bounds
    )
    offset = popt_gauss[0]
    center = popt_gauss[slice(1, n_gaussian+1)]
    width = popt_gauss[slice(n_gaussian+1, len(popt_gauss))]
    perr_gauss = np.sqrt(np.diag(pcov_gauss))
    d_offset = perr_gauss[0]
    d_center = perr_gauss[slice(1, n_gaussian+1)]
    d_width = perr_gauss[slice(n_gaussian+1, len(perr_gauss))]
    return offset, center, width, d_offset, d_center, d_width


def run_prediction(
        run_name, photon_rate_mhz=30,sampling_rate_mhz=250, batch_size=400, noise_lsb=1.05,
        bin_size_ns=0.5, n_sample=90, sigma_smooth_photon_ns=0.
):
    # toy parameters
    n_sample_init = 20
    amplitude_gain = 5.

    generator = generator_for_training(
        n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, photon_rate_mhz=photon_rate_mhz,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
        sigma_smooth_photon_ns=sigma_smooth_photon_ns
    )
    waveform, photon = next(generator)

    model = tf.keras.models.load_model(
        './Model/' + run_name + '.h5',
        custom_objects={
            'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
            'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
        }
    )
    print('model ' + run_name + ' is loaded')
    predict_photon = model.predict(waveform)
    loss = model.evaluate(x=waveform, y=photon)
    print('ĺoss=', loss)
    return waveform, photon, predict_photon,


def photon_stat(bin_size_ns, photon_truth, predict_photon):
    if photon_truth.ndim > 1:
        raise ValueError('photon_truth must be a vector')
    if predict_photon.ndim > 1:
        raise ValueError('predict_photon must be a vector')
    n_bin = len(photon_truth)
    t_photon_ns = np.arange(0, n_bin) * bin_size_ns
    t_predict = []
    temp = photon_truth.flatten().copy()
    while np.sum(temp) > 0:
        t_predict.extend(t_photon_ns[temp >= 1])
        temp[temp >= 1] -= 1
    offset, center, width, d_offset, d_center, d_width = fit_sum_norm_gaussian(
        t_photon_ns[50:-50],
        predict_photon[50:-50] / bin_size_ns,
        int(np.sum(photon_truth[50:-50])),
        cen_init=t_predict
    )
    return offset, center, width, d_offset, d_center, d_width


def plot_prediction(
        bin_size_ns, photon_truth, predict_photon, t_samples_ns, waveform,
        title=None, filename=None
):
    offset, center, width, d_offset, d_center, d_width = photon_stat(
        bin_size_ns, photon_truth[0, :], predict_photon[0, :]
    )
    n_bin = photon_truth.shape[1]
    t_photon_ns = np.arange(0, n_bin) * bin_size_ns

    fig, axes = plt.subplots(3, 1, sharex='all', figsize=(8, 8))
    axes[0].plot(t_samples_ns, waveform[0, :])
    axes[0].set_ylabel('waveform [LSB]')
    axes[1].plot(t_photon_ns, photon_truth[0, :], label="truth")
    axes[1].plot(
        t_photon_ns[50:-50],
        predict_photon[0, 50:-50],
        label="proba pred."
    )
    axes[1].plot(
        t_photon_ns[50:-50],
        sum_norm_gaussian(t_photon_ns[50:-50], offset, *center, *width) * bin_size_ns,
        label="gaussian fit",
    )
    axes[1].set_ylabel('#photons')
    axes[1].legend()
    if title is not None:
        axes[1].set_title(title)
    axes[2].plot(t_photon_ns[50:-50], np.cumsum(photon_truth[0, 50:-50]), label="truth")
    axes[2].plot(t_photon_ns[50:-50], np.cumsum(predict_photon[0, 50:-50]), label="prediction")
    axes[2].plot(
        t_photon_ns[50:-50], np.cumsum(photon_truth[0, 50:-50]) - np.cumsum(predict_photon[0, 50:-50]),
        label="difference"
    )
    axes[2].plot()
    axes[2].set_xlabel('time [ns]')
    axes[2].set_ylabel('cumulative # photons')
    axes[2].legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_probability_check(predict_photon, photon_truth,
                           title=None, filename=None):
    plt.figure(figsize=(8, 6))
    histo_pred, pred_bins = np.histogram(
        predict_photon, np.linspace(0, np.max(predict_photon),50)
    )
    histo_photon = np.zeros_like(histo_pred)
    for bin in range(len(pred_bins) - 1):
        mask_pred_bin = np.logical_and(
            pred_bins[bin] <= predict_photon,
            predict_photon < pred_bins[bin+1]
        )
        histo_photon[bin] = np.sum(photon_truth[mask_pred_bin])
    pred_bins_center = 0.5 * (pred_bins[1:] + pred_bins[:-1])
    d_pred_bins = pred_bins[1:] - pred_bins[:-1]
    plt.errorbar(
        pred_bins_center,
        histo_photon/histo_pred,
        xerr=d_pred_bins,
        yerr=np.sqrt(histo_photon)/histo_pred,
        fmt='.'
    )
    plt.plot([0, np.max(pred_bins)], [0, np.max(pred_bins)], 'k--')
    plt.xlabel('prediction value')
    plt.ylabel('realisation probability')
    plt.ylim(np.array(plt.xlim())*2)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def histo_resolution(bin_size_ns, photon_truth, predict_photon,
                     title=None, filename=None):
    batch_size = photon_truth.shape[0]
    resolutions = []
    for i in range(batch_size):
        try:
            _, _, width, _, _, d_width = photon_stat(
                bin_size_ns, photon_truth[i, :], predict_photon[i, :]
            )
        except RuntimeError:
            print('WARNING: Gaussian fit failed')
            continue
        resolutions.extend(width[d_width < 2])
    plt.figure(figsize=(8, 6))
    plt.hist(resolutions, 100)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

# plt.figure()
# plt.hist(loss, 100)
# plt.xlabel('loss')
# plt.ylabel('# of simulation')
# #plt.legend()
# plt.show()

# plt.figure()
# plt.plot(
#     np.sum(test_photon, axis=-1),
#     np.sum(test_waveform, axis=-1),
#     '+', label="true"
# )
# plt.plot(
#     np.sum(predict_photon, axis=-1),
#     np.sum(test_waveform, axis=-1),
#     '+', label='predicted'
# )
# plt.xlabel('# photons')
# plt.ylabel('sum waveform')
# plt.legend()
# plt.show()


def init(run_name):
    photon_rate_mhz = 20
    batch_size = 400
    sigma_smooth_photon_ns = 0.

    sampling_rate_mhz = 250
    noise = [0, 1, 2, 3, 4, 5]  # 1.05
    bin_size_ns = 0.5
    n_sample = 90
    for noise_lsb in noise:
        waveform, photon_truth, predict_photon = run_prediction(
            run_name, photon_rate_mhz=photon_rate_mhz,
            sampling_rate_mhz=sampling_rate_mhz, batch_size=batch_size,
            noise_lsb=noise_lsb, bin_size_ns=bin_size_ns,
            n_sample=n_sample, sigma_smooth_photon_ns=sigma_smooth_photon_ns
        )
        t_samples_ns = np.arange(0, n_sample) * 1000 / sampling_rate_mhz
        plot_prediction(
            bin_size_ns, photon_truth, predict_photon, t_samples_ns, waveform,
            filename='plots/predict_noise' + str(noise_lsb) + '.png'
        )
        plot_probability_check(
            predict_photon, photon_truth,
            filename='plots/probability_noise' + str(noise_lsb) + '.png'
        )
        histo_resolution(
            bin_size_ns, photon_truth, predict_photon,
            filename='plots/resolution_noise' + str(noise_lsb) + '.png'
        )

if __name__ == '__main__':
    init('conv_filter488_kernel10_lr0.001_LC_rate0-200_smooth2_run0rrrr')