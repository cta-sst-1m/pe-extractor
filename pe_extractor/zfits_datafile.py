from protozfits import File as ZfitsFile
import numpy as np
import sys


class ZfitsRun:
    def __init__(self, zfits_filenames, n_channel=2):
        self.zfits_filenames = zfits_filenames
        self.buffer_datafile = None
        self.n_channel = n_channel

    def get_generator_waveform(self, batch_size=1, baseline=0.):
        for f in self.zfits_filenames:
            print("Reading", f)
            if self.buffer_datafile is None:
                self.buffer_datafile = ZfitsDatafile(f, self.n_channel, n_synchro_max=batch_size)
            else:
                self.buffer_datafile.add_datafile(ZfitsDatafile(f, self.n_channel, n_synchro_max=batch_size))
            batch = self.buffer_datafile.get_batch_synchronized(
                batch_size, remove_events=True
            )
            while batch.shape[0] == batch_size:
                yield batch - baseline
                batch = self.buffer_datafile.get_batch_synchronized(
                    batch_size, remove_events=True
                )


class ZfitsDatafile:
    def __init__(self, zfits_filename, n_channel=2, n_sample=4320, n_synchro_max=None):
        self.generator = ZfitsFile(zfits_filename).Events
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
                print("WARNING: error of", event.local_time_nanosec - acq.time_ns,
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


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from pe_extractor.cnn import Correlation, Extractor

    laser_off = ZfitsRun(
        [
            "experimental_waveforms/SST1M_01_20200121_0000.fits.fz",
            "experimental_waveforms/SST1M_01_20200121_0001.fits.fz",
            "experimental_waveforms/SST1M_01_20200121_0002.fits.fz",
            "experimental_waveforms/SST1M_01_20200121_0003.fits.fz",
            "experimental_waveforms/SST1M_01_20200121_0004.fits.fz",
        ]
    )

    batch_size = 1000

    generator = laser_off.get_generator_waveform(batch_size)
    mean_fft = None
    sum_fft_per_batch = []
    n_batch = 0
    #for batch_wf in tqdm(generator, desc="batch ({:} wf/batch)".format(batch_size)):
    for batch_wf in generator:
        baseline = np.mean(np.mean(batch_wf, axis=1), axis=0)
        baseline_wf = np.tile(baseline.reshape([1,1, -1]), [batch_size, 4320, 1])
        batch_wf -= baseline_wf
        fft = np.fft.fft(batch_wf, axis=1)
        sum_fft_per_batch.append(np.sum(fft, axis=0, keepdims=True))
        if mean_fft is None:
            mean_fft = np.mean(np.abs(fft), axis=0)
        else:
            mean_fft += np.mean(np.abs(fft), axis=0)
        n_batch += 1
    n_sample = mean_fft.shape[0]
    mean_fft /= n_batch
    freq_MHz = np.fft.fftfreq(n_sample, d=4e-9)*1e-6
    sum_fft_per_batch = np.concatenate(sum_fft_per_batch, axis=0)
    phases = np.angle(sum_fft_per_batch)
    phases_diff = phases[:, :, 0] - phases[:, :, 1]
    std_phase_diff = np.std(phases_diff, axis=0)

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
    axes[1].plot(freq_MHz[freq_MHz > 0], std_phase_diff[freq_MHz > 0], '-')
    axes[1].set_ylabel('std of phase difference')
    axes[1].set_xlabel('frequency (MHz)')
    axes[1].grid()
    fig.show()

    print(
        "frequencies (MHz) with correlated noise between pixels:",
        freq_MHz[std_phase_diff < 2.3]
    )

    fig, axes = plt.subplots(1, 1)
    dfreq_MHz = freq_MHz[1] - freq_MHz[0]
    freq_histo_MHz = np.concatenate([
        freq_MHz - dfreq_MHz/2, [freq_MHz[-1] + dfreq_MHz/2]
    ])
    time_histo = np.arange(sum_fft_per_batch.shape[0] + 1)
    #im = axes[0].pcolormesh(freq_histo_MHz[freq_histo_MHz>=0], time_histo, phases[:, freq_MHz>0, 0])
    #fig.colorbar(im, ax=axes[0])
    #im = axes[1].pcolormesh(freq_histo_MHz[freq_histo_MHz>=0], time_histo, phases[:, freq_MHz>0, 1])
    #fig.colorbar(im, ax=axes[1])
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
    fig.show()