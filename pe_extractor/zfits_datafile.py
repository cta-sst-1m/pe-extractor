from protozfits import File as ZfitsFile
import numpy as np
from matplotlib import pyplot as plt
from pe_extractor.cnn import Correlation, Extractor
import tensorflow as tf
from tqdm import tqdm
from pe_extractor.toy import get_baseline
from scipy.signal import butter, lfilter, freqz
import os


class ZfitsRun:
    def __init__(
            self, zfits_filenames, n_channel=2,
            lowpass_cut_MHz=None, order_filter=8, sampling_freq_MHz=250,
    ):
        """
        Create a run from a list of zfits files. Also optionally add a lowpass
        filter.
        :param zfits_filenames: list of input zfits files
        :param n_channel: numper of pixels to synchronize
        :param lowpass_cut_MHz: cutting frequency of the lowpass filter applied
        to waveforms. Set to None to not apply any lowpass filter.
        :param order_filter: order of the Butterworth digital filter applied
        to the data
        :param sampling_freq_MHz: sampling frequency, used only if
        lowpass_cut_MHz is not None
        """
        self.zfits_filenames = []
        for f in zfits_filenames:
            if os.path.isfile(f):
                self.zfits_filenames.append(f)
            else:
                print('WARNING: file', f, 'skipped as it could not be found')
        self.buffer_datafile = None
        self.n_channel = n_channel
        self.set_lowpass_filter(
            lowpass_cut_MHz=lowpass_cut_MHz, order_filter=order_filter,
            sampling_freq_MHz=sampling_freq_MHz
        )

    def set_lowpass_filter(
            self, lowpass_cut_MHz=None, order_filter=8, sampling_freq_MHz=250
    ):
        """
        allow to change the low pass filter carateristics
        :param lowpass_cut_MHz: cutting frequency of the lowpass filter applied
        to waveforms. Set to None to not apply any lowpass filter.
        :param order_filter: order of the Butterworth digital filter applied
        to the data
        :param sampling_freq_MHz: sampling frequency, used only if
        lowpass_cut_MHz is not None
        """
        if lowpass_cut_MHz is not None:
            print('setting up low pass filter wit f_c=', lowpass_cut_MHz, 'MHz')
            self.b_poly, self.a_poly = butter(
                order_filter, 2*lowpass_cut_MHz/sampling_freq_MHz,
                btype='low', analog=False
            )
        else:
            print('removing the low pass filter')
            self.a_poly = None
            self.b_poly = None

    def get_generator_waveform(
            self, batch_size=1, baselines=0., full_batch=True,
            synchro_between_files=False
    ):
        """
        function returning a generator yielding a batch of baseline-subtracted
        waveforms from all files given to the contructor of ZfitsRun.
        A low pass filter is applied to the waverform.
        :param batch_size: number of waveforms to be returned per per batch
        :param baselines: value of the baseline that will be subtracted for each pixel
        :param full_batch: if True, only full batches are returned.
        :param synchro_between_files: if True the event form the different channels will
        be synchronized between files, otherwise the events are synchronized only within
        a file.
        :return: a generator of batch of waveform (size n_batch x n_sample x n_pix)
        """
        baselines = np.array(baselines).reshape([1, 1, -1])
        incomplete_batch = None
        for f in self.zfits_filenames:
            print("Reading", f)
            if self.buffer_datafile is None:
                self.buffer_datafile = ZfitsDatafile(
                    f, self.n_channel, n_synchro_max=batch_size
                )
            else:
                if synchro_between_files:
                    self.buffer_datafile.add_datafile(
                        ZfitsDatafile(f, self.n_channel, n_synchro_max=batch_size)
                    )
                else:
                    self.buffer_datafile = ZfitsDatafile(
                        f, self.n_channel, n_synchro_max=batch_size
                    )
            if incomplete_batch is not None:
                new_batch = self.buffer_datafile.get_batch_synchronized(
                    batch_size - incomplete_batch.shape[0], remove_events=True
                )
                batch = np.concatenate([incomplete_batch, new_batch], axis=0)
            else:
                batch = self.buffer_datafile.get_batch_synchronized(
                    batch_size, remove_events=True
                )
            while batch.shape[0] == batch_size:
                if self.a_poly is None:
                    yield batch - baselines
                else:
                    yield lfilter(
                        self.b_poly, self.a_poly, batch - baselines, axis=1
                    )
                batch = self.buffer_datafile.get_batch_synchronized(
                    batch_size, remove_events=True
                )
            if full_batch:
                incomplete_batch = batch
            else:
                if self.a_poly is None:
                    yield batch - baselines
                else:
                    yield lfilter(
                        self.b_poly, self.a_poly, batch - baselines, axis=1
                    )
        self.reset()

    def reset(self):
        """
        function to start again from the beginning of the run
        """
        self.buffer_datafile = None

    def get_baselines(self, n_waveform=1000, margin_lsb=8, samples_around=4):
        wfs_for_baseline = next(
            self.get_generator_waveform(batch_size=n_waveform)
        )
        baselines = []
        for pix in range(self.n_channel):
            baselines.append(
                get_baseline(
                    wfs_for_baseline[:, :, pix],
                    margin_lsb=margin_lsb, samples_around=samples_around
                )
            )
            if not np.isfinite(baselines[-1]):
                print('ERROR in baseline calculation for pixel', pix)
                baselines[-1] = 0
            print('baseline for pixel', pix, ':', baselines[-1])
        self.reset()
        return np.array(baselines)


class ZfitsDatafile:
    def __init__(self, zfits_filename, n_channel=2, n_sample=4320, n_synchro_max=None):
        self._file = ZfitsFile(zfits_filename)
        self.generator = self._file.Events
        self.events = {}
        self.synchronized = []
        self.not_synchronized = []
        self.n_channel = n_channel
        self.n_sample = n_sample
        event = next(self.generator)
        while True:
            self.add_camera_event(event)
            if n_synchro_max is not None:
                if len(self.synchronized) >= n_synchro_max:
                    break
            try:
                event = next(self.generator)
            except StopIteration:
                break

    def __del__(self):
        self._file.close()
        self.events = {}
        self.synchronized = []
        self.not_synchronized = []

    def add_camera_event(self, event, disable_different_timing=True):
        evt_nb = event.eventNumber
        samples = event.sii.samples
        assert self.n_sample == np.size(samples)
        if evt_nb in self.events.keys():
            acq = self.events[evt_nb]
            if event.local_time_nanosec != acq.time_ns or event.local_time_sec != acq.time_s:
                if disable_different_timing:
                    return
                print("WARNING: error of", event.local_time_nanosec - acq.time_ns,
                      "ns synchronizing event", evt_nb)
            acq.set_channel(event.sii.channelId, samples)
            nrecorded_channel = len(acq.filled_channels)
            if nrecorded_channel == self.n_channel:
                self.not_synchronized.remove(evt_nb)
                self.synchronized.append(evt_nb)
            elif nrecorded_channel == 1:
                self.not_synchronized.append(evt_nb)
        else:
            self.events[evt_nb] = SiiAcquisition(
                time_sec=event.local_time_sec,
                time_nanosec=event.local_time_nanosec,
                channel=event.sii.channelId,
                wf=samples,
                n_channel=self.n_channel
            )
            self.not_synchronized.append(evt_nb)

    def remove(self, evt_nb):
        self.events.pop(evt_nb)
        if evt_nb in self.synchronized:
            self.synchronized.remove(evt_nb)
        else:
            self.not_synchronized.remove(evt_nb)

    def add_sii_acquisition(self, evt_nb, event, disable_different_timing=True):
        assert isinstance(event, SiiAcquisition)
        if evt_nb in self.events.keys():
            existing = self.events[evt_nb]
            if event.time_ns != existing.time_ns or event.time_s != existing.time_s:
                if disable_different_timing:
                    return
                print("WARNING: error of", event.time_ns - existing.time_ns,
                      "ns synchronizing event", evt_nb)
            for channel in event.filled_channels:
                existing.set_channel(channel, event.wf[channel])
            if existing.is_synchronized():
                self.not_synchronized.remove(evt_nb)
                self.synchronized.append(evt_nb)
        else:
            first_filled_channel = event.filled_channels[0]
            self.events[evt_nb] = SiiAcquisition(
                time_sec=event.time_s,
                time_nanosec=event.time_ns,
                channel=first_filled_channel,
                wf=event.wf[first_filled_channel, :],
                n_channel=self.n_channel
            )
            for channel in event.filled_channels[1:]:
                self.events[evt_nb].set_channel(channel, event.wf[channel, :])
            if self.events[evt_nb].is_synchronized():
                self.synchronized.append(evt_nb)
            else:
                self.not_synchronized.append(evt_nb)

    def add_datafile(self, datafile):
        self.generator = datafile.generator
        for evt_nb in datafile.events.keys():
            event = datafile.events[evt_nb]
            self.add_sii_acquisition(evt_nb, event)

    def get_batch_synchronized(self, batch_size, remove_events=False):
        synchronized = self.synchronized
        if len(synchronized) < batch_size:
            event = next(self.generator)
            while len(self.synchronized) < batch_size:
                self.add_camera_event(event)
                try:
                    event = next(self.generator)
                except StopIteration:
                    break
        if len(synchronized) < batch_size:
            print("WARNING: not enough synchronized events to make a full batch")
            batch_waveform = np.zeros(
                [len(synchronized), self.n_sample, self.n_channel])
        else:
            batch_waveform = np.zeros(
                [batch_size, self.n_sample, self.n_channel])
        event_read = []
        for batch_idx, evt_nb in enumerate(synchronized):
            if batch_idx == batch_size:
                break
            batch_waveform[batch_idx, :, :] = self.events[evt_nb].wf.T
            event_read.append(evt_nb)
        if remove_events:
            for evt_nb in event_read:
                self.remove(evt_nb)
        return batch_waveform


class SiiAcquisition:
    def __init__(self, time_sec, time_nanosec, channel, wf, n_channel=2):
        self.time_s = time_sec
        self.time_ns = time_nanosec
        n_sample = np.size(wf)
        self.wf = np.zeros([n_channel, n_sample])
        self.wf[channel, :] = wf
        self.filled_channels = [channel]
        self.n_channel = n_channel

    def set_channel(self, channel, wf):
        if channel in self.filled_channels:
            print("WARNING: overwriting channel", channel)
        else:
            self.filled_channels.append(channel)
        self.wf[channel, :] = wf

    def is_synchronized(self):
        return len(self.filled_channels) == self.n_channel


def get_fft_avg(zfits_run, batch_size=1000):
    generator_sii = zfits_run.get_generator_waveform(batch_size=batch_size)
    sum_fft_per_batch = []
    n_waveform = 0
    n_sample = 0
    mean_fft = None
    pbar = tqdm(desc='wf')
    for batch_wf in generator_sii:
        batch_size, n_sample, _ = batch_wf.shape
        pbar.update(batch_size)
        baseline = np.mean(np.mean(batch_wf, axis=1), axis=0)
        baseline_wf = np.tile(baseline.reshape([1,1, -1]), [batch_size, 4320, 1])
        batch_wf -= baseline_wf
        fft = np.fft.rfft(batch_wf, axis=1, norm='ortho')
        sum_fft_per_batch.append(np.sum(fft, axis=0, keepdims=True))
        if mean_fft is None:
            mean_fft = np.sum(np.abs(fft), axis=0)
        else:
            mean_fft += np.sum(np.abs(fft), axis=0)
        n_waveform += batch_size
    pbar.close()
    mean_fft /= n_waveform
    freq_MHz = np.fft.rfftfreq(n_sample, d=4e-9)*1e-6
    sum_fft_per_batch = np.concatenate(sum_fft_per_batch, axis=0)
    phases_per_batch = np.angle(sum_fft_per_batch)
    zfits_run.reset()
    return freq_MHz, mean_fft, phases_per_batch


def plot_fft_avg(zfits_run, batch_size=1000, title=None, plot=None):
    freq_MHz, mean_fft, phases_per_batch = get_fft_avg(
        zfits_run, batch_size=batch_size
    )
    phases_diff = phases_per_batch[:, :, 0] - phases_per_batch[:, :, 1]
    std_phase_diff = np.std(phases_diff, axis=0)
    freq_coherent_MHz = freq_MHz[std_phase_diff < 2.3]

    if plot is not None:
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].semilogy(
            freq_MHz[freq_MHz >= 0], mean_fft[freq_MHz >= 0, 0],
            '-', label='pixel 0'
        )
        axes[0].semilogy(
            freq_MHz[freq_MHz >= 0], mean_fft[freq_MHz >= 0, 0],
            '--', label='pixel 1')
        axes[0].set_ylabel('amplitude of fft')
        axes[0].set_xlim([0, np.max(freq_MHz)])
        axes[0].grid()
        axes[0].legend()
        if title is not None:
            axes[0].set_title(title)
        axes[1].plot(freq_MHz[freq_MHz > 0], std_phase_diff[freq_MHz > 0], '-')
        axes[1].set_ylabel('std of phase difference')
        axes[1].set_xlabel('frequency (MHz)')
        axes[1].grid()
        if plot.lower() == "show":
            fig.show()
        else:
            plt.savefig(plot)
        plt.close(fig)
    return freq_coherent_MHz


def plot_phase_diff_evolution(zfits_run, batch_size=1000, title=None):
    freq_MHz, mean_fft, phases_per_batch = get_fft_avg(
        zfits_run, batch_size=batch_size
    )
    phases_diff = phases_per_batch[:, :, 0] - phases_per_batch[:, :, 1]
    fig, axes = plt.subplots(1, 1)
    dfreq_MHz = freq_MHz[1] - freq_MHz[0]
    freq_histo_MHz = np.concatenate([
        freq_MHz - dfreq_MHz/2, [freq_MHz[-1] + dfreq_MHz/2]
    ])
    time_histo = np.arange(phases_diff.shape[0] + 1)
    im = plt.pcolormesh(
        freq_histo_MHz[freq_histo_MHz >= 0],
        time_histo,
        phases_diff[:, freq_MHz > 0],
        cmap=plt.get_cmap('hsv'), vmin=-np.pi, vmax=np.pi
    )
    cbar = fig.colorbar(im)
    cbar.set_label('angle difference of fft')
    plt.xlim([55, 70])
    plt.xlabel('frequency (MHz)')
    if title is not None:
        plt.title(title)
    fig.show()


def g2_from_files(zfits_run, shift_bin, title=None, batch_size=1000):

    correlator = Correlation(
        shifts=shift_bin, n_batch=batch_size, n_sample=4320,
        sample_type=tf.float64, scope="wf", parallel_iterations=100
    )
    baselines = zfits_run.get_baselines()
    generator_sii = zfits_run.get_generator_waveform(
        batch_size=batch_size, baselines=baselines
    )

    sum_1_all = 0
    sum_2_all = 0
    sum_11_all = 0
    sum_12_all = 0
    sum_22_all = 0
    count_all = 0
    pbar = tqdm(desc='wf')
    while True:
        try:
            wf_pixels = next(generator_sii)
        except StopIteration:
            break
        sum_1, sum_2, sum_11, sum_12, sum_22, count = correlator(
            wf_pixels[:, :, 0], wf_pixels[:, :, 1]
        )
        pbar.update(wf_pixels.shape[0])
        sum_1_all += sum_1
        sum_2_all += sum_2
        sum_11_all += sum_11
        sum_12_all += sum_12
        sum_22_all += sum_22
        count_all += count
    pbar.close()
    g2_12 = count_all * sum_12_all / (sum_1_all * sum_2_all)
    time_shift_ns = shift_bin * 4
    plt.plot(time_shift_ns, g2_12, 'b-+', label='no filter')
    plt.xlabel('time (ns)')
    plt.ylabel(r'$g^2$')
    plt.grid()
    if title is not None:
        plt.title(title)
    plt.show()
    zfits_run.reset()


def get_psd(zfits_run, batch_size=1000):
    generator_sii = zfits_run.get_generator_waveform(batch_size=batch_size, )
    n_waveform = 0
    psd_1 = None
    psd_2 = None
    csd = None
    pbar = tqdm(desc='wf')
    n_sample = 0
    for batch_wf in generator_sii:
        batch_size, n_sample, _ = batch_wf.shape
        pbar.update(batch_size)
        baseline = np.mean(np.mean(batch_wf, axis=1), axis=0)
        baseline_wf = np.tile(baseline.reshape([1,1, -1]), [batch_size, 4320, 1])
        batch_wf -= baseline_wf
        fft = np.fft.rfft(batch_wf, axis=1, norm='ortho')
        if psd_1 is None:
            psd_1 = np.sum(fft[:, :, 0] * np.conj(fft[:, :, 0]), axis=0)
            psd_2 = np.sum(fft[:, :, 1] * np.conj(fft[:, :, 1]), axis=0)
            csd = np.sum(np.conj(fft[:, :, 0])*fft[:, :, 1], axis=0)
        else:
            psd_1 += np.sum(fft[:, :, 0] * np.conj(fft[:, :, 0]), axis=0)
            psd_2 += np.sum(fft[:, :, 1] * np.conj(fft[:, :, 1]), axis=0)
            csd += np.sum(np.conj(fft[:, :, 0])*fft[:, :, 1], axis=0)
        n_waveform += batch_size
    pbar.close()
    psd_1 /= n_waveform
    psd_2 /= n_waveform
    csd /= n_waveform
    freq_MHz = np.fft.rfftfreq(n_sample, d=4e-9)*1e-6
    zfits_run.reset()
    return freq_MHz, psd_1, psd_2, csd


def coherence(zfits_run, batch_size=1000, plot=None):
    freq_MHz, psd_1, psd_2, csd = get_psd(zfits_run, batch_size=batch_size)
    coher = (np.abs(csd) ** 2) / (psd_1 * psd_2)
    if plot:
        plt.semilogy(freq_MHz, psd_1, label='$psd_1$')
        plt.semilogy(freq_MHz, psd_2, label='$psd_2$')
        plt.semilogy(freq_MHz, np.abs(csd), label='$csd_{12}$')
        plt.xlabel('frequency [MHz]')
        plt.legend()
        plt.ylim([1e-4, 1e4])
        plt.show()
    return freq_MHz, np.abs(coher)


if __name__ == '__main__':
    pp_files = [
        "experimental_waveforms/SST1M_01_20200121_0006.fits.fz",
        "experimental_waveforms/SST1M_01_20200121_0007.fits.fz",
        "experimental_waveforms/SST1M_01_20200121_0008.fits.fz",
        "experimental_waveforms/SST1M_01_20200121_0009.fits.fz",
    ]
    #pp_files = [ "/data/SII/2020/02/24/SST1M_01/SST1M_01_20200224_{:04}.fits.fz".format(i) for i in range(29, 39) ] #range(29, 136)
    pp_run = ZfitsRun(pp_files)

    freq_pp, coher_pp = coherence(pp_run, batch_size=1000)
    del pp_run

    sp_files = [
        "experimental_waveforms/SST1M_01_20200121_0353.fits.fz",
        "experimental_waveforms/SST1M_01_20200121_0354.fits.fz",
        "experimental_waveforms/SST1M_01_20200121_0355.fits.fz",
        "experimental_waveforms/SST1M_01_20200121_0356.fits.fz",
    ]
    #sp_files = [ "/data/SII/2020/02/24/SST1M_01/SST1M_01_20200224_{:04}.fits.fz".format(i) for i in range(528, 538) ] #range(528, 614)
    sp_run = ZfitsRun(sp_files)

    freq_sp, coher_sp = coherence(sp_run, batch_size=1000)
    del sp_run

    fig, axes = plt.subplots(2, 1, sharex='col')
    axes[0].semilogy(freq_pp, coher_pp)
    axes[0].set_title('coherence PP run')
    axes[0].set_xlabel('frequency [MHz]')
    axes[0].set_xlim([freq_pp[0], freq_pp[-1]])
    axes[0].set_ylim([1e-6, 1])
    axes[0].grid()
    axes[1].semilogy(freq_sp, coher_sp)
    axes[1].set_title('coherence SP run')
    axes[1].set_xlabel('frequency [MHz]')
    axes[0].set_xlim([freq_sp[0], freq_sp[-1]])
    axes[1].set_ylim([1e-6, 1])
    axes[1].grid()
    plt.tight_layout()
    plt.savefig('coherence_20200121.png')
    #plt.savefig('coherence_20200224.png')
