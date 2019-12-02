from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pe_extractor.toy import generator_nsb, generator_flash, generator_andrii_toy
from pe_extractor.train_cnn import loss_all, loss_cumulative, loss_chi2, loss_continuity, \
    timebin_from_prediction
import tensorflow as tf
import os
from .intensity_interferometry import get_baseline
from .toy import read_experimental, model_predict


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


def sum_gaussian(x_val, offset, amplitudes, centers, widths):
    n_gauss = np.size(amplitudes)
    n_val = np.size(x_val)
    gauss_cen = np.tile(np.array(centers).reshape([n_gauss, 1]), [1, n_val])
    gauss_width = np.tile(np.array(widths).reshape([n_gauss, 1]), [1, n_val])
    gauss_amplitude = np.tile(
        np.array(amplitudes).reshape([n_gauss, 1]),
        [1, n_val]
    )
    gauss_norm = np.sqrt(2 * np.pi) * gauss_width
    gauss_x = np.tile(
        np.array(x_val).reshape([1, -1]),
        [n_gauss, 1]
    )
    amp_gaussians = gauss_amplitude / gauss_norm * np.exp(
        - 0.5 * ((gauss_x - gauss_cen) / gauss_width) ** 2
    )
    return np.sum(amp_gaussians, axis=0) + offset


def fit_gaussian_pulse(
    x_array, y_array, amplitudes_init, centers_init, widths_init, offset_init=0.
):
    n_gaussian = len(amplitudes_init)
    if n_gaussian != len(centers_init) or n_gaussian != len(widths_init):
        raise ValueError(
            'amplitudes, centers_init and widths_init must be of same length.'
        )

    p0 = np.array([offset_init, amplitudes_init, centers_init, widths_init]).flatten()
    offset_min = -0.01 * np.ones(n_gaussian)
    offset_max = 0.01 * np.ones(n_gaussian)
    amplitudes_min = np.zeros(n_gaussian)
    amplitudes_max = 3e4 * np.ones(n_gaussian)
    centers_min = (x_array[0] - 10.) * np.ones(n_gaussian)
    centers_max = (x_array[-1] + 10.) * np.ones(n_gaussian)
    widths_min = 0.5 * np.ones(n_gaussian)
    widths_max = 10 * np.ones(n_gaussian)
    bounds_min = np.array([offset_min, amplitudes_min, centers_min, widths_min])
    bounds_max = np.array([offset_max, amplitudes_max, centers_max, widths_max])
    bounds = (bounds_min.flatten(), bounds_max.flatten())
    popt_gauss, pcov_gauss = curve_fit(
        sum_gaussian, x_array, y_array,
        p0=p0, bounds=bounds
    )
    offset = popt_gauss[0]
    amplitude = popt_gauss[1]
    center = popt_gauss[2]
    width = popt_gauss[3]
    perr_gauss = np.sqrt(np.diag(pcov_gauss))
    d_offset = perr_gauss[0]
    d_amplitude = popt_gauss[1]
    d_center = perr_gauss[2]
    d_width = perr_gauss[3]
    return offset, amplitude, center, width, d_amplitude, d_offset, d_center, \
           d_width


def fit_sum_norm_gaussian(
        x_array, y_array, n_gaussian, cen_gauss_init=None,
):
    x_min = x_array[0] - 10.
    x_max = x_array[-1] + 10.
    if cen_gauss_init is None:
        # distribute Gaussian initial position
        gauss_cen_init = np.linspace(
            x_array[0], x_array[-1], n_gaussian + 2
        )[1:-1]
    else:
        cen_gauss_init = np.array(cen_gauss_init)
        gauss_cen_init = cen_gauss_init[cen_gauss_init >= x_min]
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
    p0.extend(4. * np.ones([n_gaussian])) # 4 sigmas for start of fit
    # give little weight to bins with y <
    sigma_array_gauss = 1/(.1 + (np.abs(y_array) >= 0.01).astype(float))
    popt_gauss, pcov_gauss = curve_fit(
        sum_norm_gaussian, x_array, y_array,
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


def toy_nsb_prediction(
        run_name, pe_rate_mhz=30, sampling_rate_mhz=250, batch_size=400,
        noise_lsb=1.05, bin_size_ns=0.5, n_sample=90, sigma_smooth_pe_ns=0.,
        baseline=0., relative_gain_std=0.1, shift_proba_bin=0
):
    # toy parameters
    n_sample_init = 50
    amplitude_gain = 5.

    generator = generator_nsb(
        n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
        n_sample_init=n_sample_init, pe_rate_mhz=pe_rate_mhz,
        bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
        amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
        sigma_smooth_pe_ns=sigma_smooth_pe_ns, baseline=baseline,
        relative_gain_std=relative_gain_std, shift_proba_bin=shift_proba_bin
    )
    waveform, pe = next(generator)

    model = tf.keras.models.load_model(
        './Model/' + run_name + '.h5',
        custom_objects={
            'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
            'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
        }
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_all,  # loss_all
        metrics=[loss_cumulative, loss_chi2, loss_continuity]  # loss_cumulative, loss_chi2, loss_continuity
    )
    print('model ' + run_name + ' is loaded')
    predict_pe = model.predict(waveform)
    loss = model.evaluate(x=waveform, y=pe)
    print('ĺoss=', loss)
    return waveform, pe, predict_pe,


def pe_stat(bin_size_ns, pe_truth, predict_pe, fit_type="norm_gauss"):
    if pe_truth.ndim > 1:
        raise ValueError('pe_truth must be a vector')
    if predict_pe.ndim > 1:
        raise ValueError('predict_pe must be a vector')
    n_bin = len(pe_truth)
    t_pe_ns = np.arange(0, n_bin) * bin_size_ns
    if fit_type == "norm_gauss":
        t_predict = []
        temp = pe_truth.flatten().copy()
        while np.sum(temp) > 0:
            t_predict.extend(t_pe_ns[temp >= 1])
            temp[temp >= 1] -= 1
        offset, center, width, d_offset, d_center, d_width \
            = fit_sum_norm_gaussian(
                t_pe_ns[50:-50],
                predict_pe[50:-50] / bin_size_ns,
                int(np.sum(pe_truth[50:-50])),
                cen_gauss_init=t_predict
            )
        amplitude = np.ones_like(center)
        d_amplitude = np.zeros_like(amplitude)
    elif fit_type == "gauss":
        with_pe = pe_truth >= 1
        centers_init = t_pe_ns[with_pe]
        ampl_init = pe_truth[with_pe]
        widths_init = 4 * np.ones(np.sum(with_pe))
        offset_init = 0 * np.ones(np.sum(with_pe))
        (
            offset, amplitude, center, width, d_offset, d_amplitude, d_center,
            d_width
        ) = fit_gaussian_pulse(
                t_pe_ns[50:-50],
                predict_pe[50:-50] / bin_size_ns,
                amplitudes_init=ampl_init, centers_init=centers_init,
                widths_init=widths_init, offset_init=offset_init
            )
    else:
        raise ValueError('fit_type must be either "gauss" or "norm_gauss"')
    return (
        offset, amplitude, center, width, d_offset, d_amplitude, d_center,
        d_width
    )


def plot_prediction(
        bin_size_ns, pe_truth, predict_pe, t_samples_ns, waveform,
        title=None, filename=None, fit_type="norm_gauss", batch_index=0,
):
    if fit_type is not None:
        (
            offset, amplitude, center, width, d_offset, d_amplitude, d_center,
            d_width
        ) = pe_stat(
            bin_size_ns, pe_truth[batch_index, :], predict_pe[batch_index, :], fit_type=fit_type
        )

    n_bin = pe_truth.shape[1]
    t_pe_ns = np.arange(0, n_bin) * bin_size_ns + t_samples_ns[0]

    fig, axes = plt.subplots(3, 1, sharex='all', figsize=(8, 8))
    axes[0].plot(t_samples_ns, waveform[batch_index, :])
    axes[0].set_ylabel('waveform [LSB]')
    axes[1].plot(t_pe_ns, pe_truth[batch_index, :], label="truth")
    axes[1].plot(
        t_pe_ns,
        predict_pe[batch_index, :],
        label="probability predicted"
    )
    if fit_type == "norm_gauss":
        axes[1].plot(
            t_pe_ns,
            bin_size_ns * sum_norm_gaussian(
                t_pe_ns, offset, *center, *width
            ),
            '--', label="gaussian fit",
        )
    elif fit_type == "gauss":
        axes[1].plot(
            t_pe_ns,
            bin_size_ns * sum_gaussian(
                t_pe_ns, offset, amplitude, center, width
            ),
            '--', label="gaussian fit",
        )
    axes[1].set_ylabel('# pe')
    axes[1].legend()
    axes[2].plot(
        t_pe_ns, np.cumsum(pe_truth[batch_index, :]),
        label="truth"
    )
    axes[2].plot(
        t_pe_ns, np.cumsum(predict_pe[batch_index, :]),
        label="prediction"
    )
    axes[2].plot(
        t_pe_ns,
        np.cumsum(pe_truth[batch_index, :]) - np.cumsum(predict_pe[batch_index, :]),
        label="truth - prediction"
    )
    axes[2].set_xlabel('time [ns]')
    axes[2].set_ylabel('cumulative # pe')
    axes[2].legend()
    if title is not None:
        axes[0].set_title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        print(filename, 'created')
    else:
        plt.show()
    plt.close(fig)


def plot_probability_check(predict_pe, pe_truth, n_bin=50,
                           title=None, filename=None, binning='adaptive'):
    # predict_pe = np.abs(predict_pe, 0)
    fig = plt.figure(figsize=(8, 6))
    pred_bin_min = np.min(predict_pe) - .001
    pred_bin_max = np.max(predict_pe) + .001
    if binning == 'adaptive':
        pe = pe_truth.flatten()
        pred = predict_pe.flatten()
        # there must be the same number of true pes in each bin.
        n_pe = np.sum(pe)
        n_pe_bins = np.linspace(0, n_pe, n_bin)
        order_prediction = np.argsort(pred)
        cumsum_pe_ordered_pred = np.cumsum(pe[order_prediction])
        pred_bins = [pred_bin_min, ]
        for bin_index in range(n_bin - 1):
            index_limit = np.searchsorted(
                cumsum_pe_ordered_pred, n_pe_bins[bin_index]
            )
            pred_bins.append(pred[order_prediction[index_limit]])
        pred_bins.append(pred_bin_max)
        pred_bins = np.array(pred_bins)
    elif binning == 'constant_size':
        pred_bins = np.linspace(pred_bin_min, pred_bin_max, n_bin)
    else:
        raise ValueError(
            'binning must be either \'adaptive\' or \'constant_size\'.'
        )
    histo_pe = np.zeros(n_bin)
    histo_pred = np.zeros(n_bin)
    for bin_index in range(len(pred_bins) - 1):
        mask_pred_bin = np.logical_and(
            pred_bins[bin_index] <= predict_pe,
            predict_pe < pred_bins[bin_index+1]
        )
        histo_pe[bin_index] = np.sum(pe_truth[mask_pred_bin])
        histo_pred[bin_index] = np.sum(mask_pred_bin)
    pred_bins_center = 0.5 * (pred_bins[1:] + pred_bins[:-1])
    d_pred_bins = pred_bins[1:] - pred_bins[:-1]
    plt.errorbar(
        pred_bins_center,
        histo_pe/histo_pred,
        xerr=d_pred_bins/2,
        yerr=np.sqrt(histo_pe)/histo_pred,
        fmt='.'
    )
    plt.plot([0, pred_bin_max], [0, pred_bin_max], 'k--')
    plt.xlabel('prediction value')
    plt.ylabel('realisation probability')
    plt.ylim([-1e-2, plt.ylim()[1]])
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
        print(filename, 'created')
    else:
        plt.show()
    plt.close(fig)


def time_from_prediction(y_pred, bin_size_ns=0.5):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape([1, -1])
    cum_pe_prob = np.cumsum(y_pred, axis=-1)
    samples = np.arange(y_pred.shape[1])
    time_pe = []
    for b in range(y_pred.shape[0]):
        integer_n_pe = np.arange(0.5, np.round(cum_pe_prob[b, -1]) + 0.5)
        pe_samples = np.interp(integer_n_pe, cum_pe_prob[b, :], samples)
        time_pe.append((pe_samples * bin_size_ns).tolist())
    return time_pe


def time_resolution_flash(
    predict_pe, time_flash, bin_size_ns=0.5, delta_time_ns=4.
):
    batch_size = predict_pe.shape[0]
    n_bin_around = int(np.round(delta_time_ns / bin_size_ns))
    bin_mean = int(np.round(time_flash / bin_size_ns))
    bin_min = bin_mean - n_bin_around
    bin_max = bin_mean + n_bin_around + 1
    predict_cumsum = np.cumsum(predict_pe[:, bin_min:bin_max], axis=1)
    time_cumsum = np.arange(bin_min, bin_max) * bin_size_ns
    pred_time_flash = np.zeros(batch_size)
    for b in range(batch_size):
        t = np.interp(
            predict_cumsum[b, -1]/2, predict_cumsum[b, :], time_cumsum
        )
        pred_time_flash[b] = t
    bias = np.nanmean(pred_time_flash - time_flash)
    resolution = np.nanstd(pred_time_flash - time_flash)
    return bias, resolution


def charge_flash(
    predict_pe, time_flash, bin_size_ns=0.5, delta_time_ns=4.
):
    batch_size = predict_pe.shape[0]
    n_bin_around = int(np.round(delta_time_ns / bin_size_ns))
    bin_mean = int(np.round(time_flash / bin_size_ns))
    bin_min = bin_mean - n_bin_around
    bin_max = bin_mean + n_bin_around + 1
    predict_charge = np.sum(predict_pe[:, bin_min:bin_max], axis=1)
    return predict_charge


def histo_resolution(bin_size_ns, pe_truth, predict_pe,
                     title=None, filename=None, fit_type="norm_gauss"):
    batch_size = pe_truth.shape[0]
    true_time = time_from_prediction(pe_truth, bin_size_ns=bin_size_ns)
    pred_time = time_from_prediction(predict_pe, bin_size_ns=bin_size_ns)
    gauss_widths = []
    gauss_centers = []
    for i in range(batch_size):
        try:
            _, _, center, width, _, _, d_center, d_width = pe_stat(
                bin_size_ns, pe_truth[i, :], predict_pe[i, :],
                fit_type=fit_type
            )
        except RuntimeError:
            print('WARNING: Gaussian fit failed')
            continue
        gauss_widths.extend(width[d_width < 2])
        gauss_centers.extend(center[d_center < 2])
        if fit_type == "gauss":
            true_time[i] = np.mean(true_time[i])
            pred_time[i] = np.mean(pred_time[i])
    fig = plt.figure(figsize=(8, 6))
    # plt.hist(gauss_widths, 100, label='width of gaussian fit')
    plt.hist(np.array(gauss_centers) - np.array(true_time), 100,
             label='time difference between center of gaussian and truth')
    plt.hist(np.array(pred_time) - np.array(true_time), 100,
             label='time difference between predicted and truth')
    if title is not None:
        plt.title(title)
    plt.xlabel('time [ns]')
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        print(filename, 'created')
    else:
        plt.show()
    plt.close(fig)


def plot_resolution_flash(
        model_name, filename=None,
        n_pe_flashes=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
        noise_lsb=1, nsb_rates_mhz=(40,), batch_size=400,
        time_resolution_windows_ns=(16, ),
        charge_resolution_windows_ns=(28, ),
        bin_flash=80, bin_size_ns=0.5, shift_proba_bin=0

):
    from cycler import cycler
    from matplotlib import cm

    jet = cm.get_cmap('jet')
    title = 'model ' + model_name[:20] + ', ' + str(batch_size) + \
            ' flashes per light level, noise ' + str(noise_lsb) + ' LSB'
    n_nsb_rate = len(nsb_rates_mhz)
    n_flash_pe = len(n_pe_flashes)
    n_windows_time_resol = len(time_resolution_windows_ns)
    n_windows_charge_resol = len(charge_resolution_windows_ns)
    if n_windows_time_resol > 4 or n_windows_charge_resol > 4 :
        raise ValueError('Only up to 4 windows can be plotted')
    time_bias = np.zeros([n_nsb_rate, n_flash_pe, n_windows_time_resol])
    time_resolution = np.zeros_like(time_bias)
    charge_bias = np.zeros([n_nsb_rate, n_flash_pe, n_windows_charge_resol])
    charge_resolution = np.zeros_like(charge_bias)
    model = tf.keras.models.load_model(
        './Model/' + model_name + '.h5',
        custom_objects={
            'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
            'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
        }
    )
    for i, n_pe_flash in enumerate(n_pe_flashes):
        print(n_pe_flash, 'pe flashes')
        gen_flash = generator_flash(
            n_event=1, batch_size=batch_size, n_sample=4320, bin_flash=bin_flash,
            n_pe_flash=(n_pe_flash, n_pe_flash), bin_size_ns=bin_size_ns,
            sampling_rate_mhz=250, amplitude_gain=5., noise_lsb=noise_lsb,
            shift_proba_bin=shift_proba_bin
        )
        waveform_flash, pe_truth_flash = next(gen_flash)
        for r, nsb_rate_mhz in enumerate(nsb_rates_mhz):
            gen_nsb = generator_nsb(
                n_event=1, batch_size=batch_size, n_sample=4340, n_sample_init=20,
                pe_rate_mhz=nsb_rate_mhz, bin_size_ns=bin_size_ns, sampling_rate_mhz=250,
                amplitude_gain=5., noise_lsb=0, sigma_smooth_pe_ns=0.,
                shift_proba_bin = shift_proba_bin
            )
            waveform_nsb, pe_truth_nsb = next(gen_nsb)
            predict_pe = model.predict(waveform_flash + waveform_nsb)
            for t in range(n_windows_time_resol):
                delta_time_ns = time_resolution_windows_ns[t] / 2
                t_bias, t_resol = time_resolution_flash(
                    predict_pe=predict_pe, time_flash=bin_flash*bin_size_ns,
                    bin_size_ns=bin_size_ns, delta_time_ns=delta_time_ns
                )
                time_bias[r, i, t] = t_bias
                time_resolution[r, i] = t_resol
            for c in range(n_windows_charge_resol):
                delta_time_ns = charge_resolution_windows_ns[c] / 2
                charge_pred = charge_flash(
                    predict_pe=predict_pe, time_flash=(bin_flash+shift_proba_bin)*bin_size_ns,
                    bin_size_ns=bin_size_ns, delta_time_ns=delta_time_ns
                )
                charge_true = charge_flash(
                    predict_pe=pe_truth_flash+pe_truth_nsb, time_flash=(bin_flash+shift_proba_bin)*bin_size_ns,
                    bin_size_ns=bin_size_ns, delta_time_ns=delta_time_ns
                )
                charge_bias[r, i, c] = np.nanmean(charge_pred - charge_true) /\
                                       np.mean(charge_true)
                charge_resolution[r, i, c] = np.nanstd(
                    charge_pred - charge_true
                ) / np.mean(charge_true)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].set_title('time resolution\n' + title)
    lines_rate = []
    legend_rate = []
    lines_window = []
    legend_window = []
    cc = (cycler(color=jet(np.linspace(0, 1, n_nsb_rate))) *
          cycler(linestyle=['-', '--', '-.', ':'][:n_windows_time_resol]))
    axes[0].set_prop_cycle(cc)
    axes[1].set_prop_cycle(cc)
    for r, nsb_rate_mhz in enumerate(nsb_rates_mhz):
        for t in range(n_windows_time_resol):
            l, = axes[0].semilogx(n_pe_flashes, time_bias[r, :, t])
            axes[1].loglog(n_pe_flashes, time_resolution[r, :, t])
            if r == 0:
                label = 'window ' + str(time_resolution_windows_ns[t]) + ' ns'
                lines_window.append(l)
                legend_window.append(label)
            if t == 0:
                label = 'nsb rate ' + str(nsb_rate_mhz) + ' MHz'
                lines_rate.append(l)
                legend_rate.append(label)
    axes[1].set_xlabel('# photo-electrons per flash')
    axes[0].set_ylabel('bias [ns]')
    axes[1].set_ylabel('time resolution [ns]')
    axes[1].set_ylim([1e-3, 10])
    axes[0].legend(lines_rate, legend_rate)
    axes[1].legend(lines_window, legend_window)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        saved = 'plots/time_resolution_' + filename
        plt.savefig(saved)
        print(saved, 'saved')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    lines_rate = []
    legend_rate = []
    lines_window = []
    legend_window = []
    cc = (cycler(color=jet(np.linspace(0, 1, n_nsb_rate))) *
          cycler(linestyle=['-', '--', '-.', ':'][:n_windows_charge_resol]))
    axes[0].set_prop_cycle(cc)
    axes[1].set_prop_cycle(cc)
    axes[0].set_title('charge resolution\n' + title)
    for r, nsb_rate_mhz in enumerate(nsb_rates_mhz):
        for c in range(n_windows_charge_resol):
            label = 'nsb rate ' + str(nsb_rate_mhz) + ' MHz, window ' + \
                    str(charge_resolution_windows_ns[c]) + ' ns'
            l, = axes[0].semilogx(
                n_pe_flashes, charge_bias[r, :, c] * 100, label=label
            )
            axes[1].semilogx(
                n_pe_flashes, charge_resolution[r, :, c] * 100
            )
            if r == 0:
                label = 'window ' + str(charge_resolution_windows_ns[c]) + ' ns'
                lines_window.append(l)
                legend_window.append(label)
            if c == 0:
                label = 'nsb rate ' + str(nsb_rate_mhz) + ' MHz'
                lines_rate.append(l)
                legend_rate.append(label)
    # axes[0].xlabel('# photo-electrons per flash')
    axes[1].set_xlabel('# photo-electrons per flash')
    axes[0].set_ylabel('bias [%]')
    axes[1].set_ylabel('charge resolution [%]')
    #axes[1].set_ylim([0.05, 5])
    axes[0].legend(lines_rate, legend_rate)
    axes[1].legend(lines_window, legend_window)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        saved = 'plots/charge_resolution_' + filename
        plt.savefig(saved)
        print(saved, 'saved')
    plt.close(fig)


def demo_nsb(run_name, n_sample=90, shift_proba_bin=0, batch_index=0, sample_range=(0,100), sigma_smooth_pe_ns=2):
    pe_rate_mhz = 10
    batch_size = 1
    baseline = 0.
    relative_gain_std = .05
    sampling_rate_mhz = 250
    noise = [0, 1, 2]  # 1.05
    bin_size_ns = 0.5


    t_samples_ns = np.arange(sample_range[0], sample_range[1]) * 1000 / sampling_rate_mhz
    for noise_lsb in noise:
        title = 'p.e. rate ' + str(pe_rate_mhz) + 'MHz, noise ' + \
                str(noise_lsb) + 'LSB ' + \
                str(shift_proba_bin*bin_size_ns) + ' ns delay on p.e.'
        waveform, pe_truth, predict_pe = toy_nsb_prediction(
            run_name, pe_rate_mhz=pe_rate_mhz,
            sampling_rate_mhz=sampling_rate_mhz, batch_size=batch_size,
            noise_lsb=noise_lsb, bin_size_ns=bin_size_ns,
            n_sample=n_sample, sigma_smooth_pe_ns=sigma_smooth_pe_ns,
            baseline=baseline, relative_gain_std=relative_gain_std,
            shift_proba_bin=shift_proba_bin
        )
        waveform = waveform[:, sample_range[0]:sample_range[1]]
        n_bin_per_sample = int(1000 / sampling_rate_mhz / bin_size_ns)
        bin_range=(sample_range[0]*n_bin_per_sample, sample_range[1]*n_bin_per_sample)
        pe_truth = pe_truth[:, bin_range[0]:bin_range[1]]
        predict_pe = predict_pe[:, bin_range[0]:bin_range[1]]
        directory_plot = 'plots/' + run_name
        try:
            os.makedirs(directory_plot)
        except FileExistsError:
            pass
        plot_prediction(
            bin_size_ns, pe_truth, predict_pe, t_samples_ns, waveform,
            filename=directory_plot + '/predict_noise' +
                     str(noise_lsb) + '_range' + str(sample_range[0]) +
                     '-' +  str(sample_range[1]) + '.png',
            title=title, fit_type=None, batch_index=batch_index,
        )
        # plot_probability_check(
        #     predict_pe, pe_truth,
        #     filename='plots/probability_noise' + str(noise_lsb) + '.png',
        #     title=title
        # )
        # histo_resolution(
        #     bin_size_ns, pe_truth, predict_pe,
        #     filename='plots/resolution_noise' + str(noise_lsb) + '.png',
        #     title=title
        # )


def demo_data(run_name, datafile, shift_proba_bin=0, batch_index=0, sample_range=(0,100)):
    wf0, wf1, _, _ = read_experimental(datafile, start=0, stop=1000)
    sampling_rate_mhz = 250
    bin_size_ns = 0.5
    t_samples_ns = np.arange(sample_range[0], sample_range[1]) * 1000 / sampling_rate_mhz
    title = datafile
    import tensorflow as tf
    model = tf.keras.models.load_model(
        './Model/' + run_name + '.h5',
        custom_objects={
            'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
            'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
        }
    )
    proba0 = model_predict(model, wf0, skip_bins=0)

    waveform = wf0[:, sample_range[0]:sample_range[1]]
    n_bin_per_sample = int(1000 / sampling_rate_mhz / bin_size_ns)
    bin_range=(sample_range[0]*n_bin_per_sample, sample_range[1]*n_bin_per_sample)
    predict_pe = proba0[:, bin_range[0]:bin_range[1]]
    directory_plot = 'plots/' + run_name
    try:
        os.makedirs(directory_plot)
    except FileExistsError:
        pass
    plot_prediction(
        bin_size_ns, None, predict_pe, t_samples_ns, waveform,
        filename=directory_plot + '/predict_noise' +
                 str(noise_lsb) + '_range' + str(sample_range[0]) +
                 '-' +  str(sample_range[1]) + '.png',
        title=title, fit_type=None, batch_index=batch_index,
    )


def demo_flasher(
        run_name, bin_size_ns=0.5, n_pe_flash=10, sampling_rate_mhz=250,
        n_sample=90, noise_lsb=1, batch_size=400, sample_range=(0, 80),
        bin_flash=80, shift_proba_bin=0
):
    model = tf.keras.models.load_model(
        './Model/' + run_name + '.h5',
        custom_objects={
            'loss_all': loss_all, 'loss_cumulative': loss_cumulative,
            'loss_chi2': loss_chi2, 'loss_continuity': loss_continuity
        }
    )
    print('model ' + run_name + ' is loaded')
    n_bin_per_sample = int(1000 / sampling_rate_mhz / bin_size_ns)
    bin_range = (sample_range[0]*n_bin_per_sample, sample_range[1]*n_bin_per_sample)
    gen = generator_flash(
        n_event=1, batch_size=batch_size, n_sample=n_sample, bin_flash=bin_flash,
        n_pe_flash=(n_pe_flash, n_pe_flash), bin_size_ns=bin_size_ns,
        sampling_rate_mhz=sampling_rate_mhz, amplitude_gain=5.,
        noise_lsb=noise_lsb, shift_proba_bin=shift_proba_bin
    )
    waveform, pe_truth = next(gen)
    predict_pe = model.predict(waveform)
    title = 'flash ampl.' + str(n_pe_flash) + 'pe, noise ' + \
            str(noise_lsb) + 'LSB'
    waveform = waveform[:, sample_range[0]:sample_range[1]]
    t_samples_ns = np.arange(sample_range[0], sample_range[1]) * 1000 / sampling_rate_mhz
    pe_truth = pe_truth[:, bin_range[0]:bin_range[1]]
    predict_pe = predict_pe[:, bin_range[0]:bin_range[1]]
    plot_prediction(
        bin_size_ns, pe_truth, predict_pe, t_samples_ns, waveform,
        filename='plots/predict_flash' + str(n_pe_flash) + 'pe_noise' + str(noise_lsb) + 'LSB.png',
        title=title, fit_type="gauss"
    )
    #histo_resolution(
    #    bin_size_ns, pe_truth, predict_pe,
    #    filename='plots/resolution_flash' + str(n_pe_flash) + 'pe_noise' + str(noise_lsb) + 'LSB.png',
    #    title=title, fit_type="gauss"
    #)


if __name__ == '__main__':

    #model = 'deconv_filters-16x20-8x10-4x10-2x10-1x1-1x1-1x1_lr0.0003_rel_gain_std0.1_pos_rate0-200_smooth2.0_noise0-2_baseline0_run0rr'
    model = 'C16x16_U2_C32x16_U2_C64x8_U2_C128x8_C64x4_C32x4_C16x2_C4x2_C1x1_C1x1_ns0.1_shift64_all1-50-10lr0.0002smooth1_amsgrad_run0'
    n_pe_flashes = np.unique(np.round(np.logspace(0, 3, 15)).astype(int))
    for noise_lsb in range(3):
        plot_resolution_flash(
            model, filename='noise' + str(noise_lsb) + '.png',
            n_pe_flashes=n_pe_flashes,
            noise_lsb=noise_lsb, nsb_rates_mhz=(4, 50, 250, 500),
            batch_size=1000, charge_resolution_windows_ns=(20, 28, 36),
            time_resolution_windows_ns=(8, 16, 32), shift_proba_bin=64
        )


    #demo_nsb(model, n_sample=4320, shift_proba_bin=64, batch_index=0, sample_range=(0, 4320), sigma_smooth_pe_ns=1)
    #demo_flasher(model, n_sample=4320, n_pe_flash=10, noise_lsb=1, batch_size=400, sample_range=(0, 40), shift_proba_bin=64)

