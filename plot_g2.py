#!/usr/bin/env python3
"""
plot g2 from analyzed interferometry data

Usage:
  analyze_data [options]

Options:
  -h --help                   Show this screen.
  --list_pp_files=FILE        Text file with 1 analyzed data file per line.
                              Files corresponding to parallel polarisation
                              If "none", it is not plotted. [Default: none]
  --list_sp_files=FILE        Text file with 1 analyzed data file per line
                              Files corresponding to orthogonal polarisation
                              If "none", it is not plotted. [Default: none]
  --list_off_files=FILE       Text file with 1 analyzed data file per line
                              Files without light
                              If "none", it is not plotted. [Default: none]
  --scale_pp=FLOAT            Factor applied to g2 while plotting. [Default: 1.]
  --scale_sp=FLOAT            Factor applied to g2 while plotting. [Default: 1.]
  --scale_off=FLOAT           Factor applied to g2 while plotting. [Default: 1.]
  --plot=FILE                 Plot will be saved to FILE. In the case where
                              FILE is "show", the plot is shown and not saved.
                              [Default: show]
  --title=STR                 Title of the plot. [Default: none]

"""
from docopt import docopt
import numpy as np
from matplotlib import pyplot as plt
from pe_extractor.intensity_interferometry import plot_calculated_g2, get_stat_g2


def g2_from_analyzed_files(files):
    shift_in_sample = None
    n_sample_wf = 0
    sum12_wf = 0
    sum1_wf = 0
    sum2_wf = 0
    n_sample_pb = 0
    sum12_pb = 0
    sum1_pb = 0
    sum2_pb = 0
    for f in files:
        data = np.load(f)
        if shift_in_sample is None:
            shift_in_sample = data['shift_in_sample']
        elif np.any(shift_in_sample != data['shift_in_sample']):
            print(
                'Warning: ' + f +
                ' has different shift_in_sample than previous files.' +
                ' Skipping it'
            )
            continue
        n_sample_wf += data['n_sample_wf']
        sum12_wf += data['sum12_wf']
        sum1_wf += data['sum1_wf']
        sum2_wf += data['sum2_wf']
        n_sample_pb += data['n_sample_pb']
        sum12_pb += data['sum12_pb']
        sum1_pb += data['sum1_pb']
        sum2_pb += data['sum2_pb']
        del data
    print('sum1_wf:', np.mean(sum1_wf))
    print('sum1_pb:', np.mean(sum1_pb))
    print('sum2_wf:', np.mean(sum2_wf))
    print('sum2_pb:', np.mean(sum2_pb))
    print('n_sample_wf:', np.mean(n_sample_wf))
    print('n_sample_pb:', np.mean(n_sample_pb))
    g2_wf = n_sample_wf * sum12_wf / (sum1_wf * sum2_wf)
    g2_pb = n_sample_pb * sum12_pb / (sum1_pb * sum2_pb)
    return g2_wf, g2_pb, shift_in_sample


def convert_str(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        return text


def get_history_from_analyzed_files(files):
    peak_max_wf = []
    mean_g2_wf = []
    std_g2_wf = []
    peak_max_pb = []
    mean_g2_pb = []
    std_g2_pb = []
    n_waveform_steps = []
    for f in files:
        data = np.load(f)
        peak_max_wf.append(data['peak_max_wf'])
        mean_g2_wf.append(data['mean_g2_wf'])
        std_g2_wf.append(data['std_g2_wf'])
        peak_max_pb.append(data['peak_max_pb'])
        mean_g2_pb.append(data['mean_g2_pb'])
        std_g2_pb.append(data['std_g2_pb'])
        n_waveform_steps.append(data['n_waveform_steps'])
        del data
    return peak_max_wf, mean_g2_wf, std_g2_wf, peak_max_pb, mean_g2_pb, \
           std_g2_pb, n_waveform_steps


def plot_g2_files(
        list_filenames, label=None, title=None, ax=None, color=None, scale=1.
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()
    g2_wf, g2_pb, shift_in_sample = g2_from_analyzed_files(
        list_filenames
    )
    shift_ns = shift_in_sample * 4
    peak_max, mean_g2, std_g2, mask, peak_pos = get_stat_g2(g2_wf)
    g2_ampl = peak_max - mean_g2
    label_stat = '$g^2(\delta t=' + \
                 '{}'.format(shift_ns[peak_pos]) + ')=' + \
                 '{:.1e}'.format(g2_ampl) + ' \pm ' + \
                 '{:.0e}'.format(std_g2) + '$'
    if label:
        label += ': ' + label_stat
    else:
        label = label_stat
    ax.plot(
        shift_ns, scale * (g2_wf - mean_g2 + 3*std_g2), '-', color=color, label=label
    )
    # ax.plot(
    #     shift_ns[mask], scale * (g2_wf[mask] - mean_g2 + 3*std_g2), '.', color=color, label=None
    # )
    # ax.plot(
    #     shift_ns[~mask], scale * (g2_wf[~mask] - mean_g2 + 3*std_g2), '+', color=color, label=None
    # )
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('shift [ns]')
    ax.set_ylabel('$g^2 - \mu_{baseline} + 3* \sigma_{baseline}$')
    ax.legend()
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    args = docopt(__doc__)
    list_files_pp = args['--list_pp_files']
    list_files_sp = args['--list_sp_files']
    list_files_off = args['--list_off_files']
    output_file = args['--plot']
    title = args['--title']
    scale_pp = float(args['--scale_pp'])
    scale_sp = float(args['--scale_sp'])
    scale_off = float(args['--scale_off'])
    if title.lower() == 'none':
        title = None
    ax = None
    fig = None
    if list_files_pp.lower() != 'none':
        with open(list_files_pp) as f:
            files_pp = [line.rstrip('\n') for line in f]
        label = 'PP'
        if abs(scale_pp - 1) > 1e-6:
            label += '$*{}$'.format(scale_pp)
        if len(files_pp) > 0:
            print(label)
            fig, ax = plot_g2_files(
                files_pp, label=label, color='b', ax=ax,
                scale=scale_pp
            )
    if list_files_sp.lower() != 'none':
        with open(list_files_sp) as f:
            files_sp = [line.rstrip('\n') for line in f]
        label = 'SP'
        if abs(scale_sp - 1) > 1e-6:
            label += '$*{}$'.format(scale_sp)
        if len(files_sp) > 0:
            print(label)
            fig, ax = plot_g2_files(
                files_sp, label=label, color='r', ax=ax,
                scale=scale_sp
            )
    if list_files_off.lower() != 'none':
        with open(list_files_off) as f:
            files_off = [line.rstrip('\n') for line in f]
        label = 'OFF'
        if abs(scale_off - 1) > 1e-6:
            label += '$*{}$'.format(scale_off)
        if len(files_off) > 0:
            print(label)
            fig, ax = plot_g2_files(
                files_off, label=label, color='g', ax=ax,
                scale=scale_off
            )
    if output_file.lower() == 'show':
        plt.show()
    else:
        plt.savefig(output_file)
        print('plot', output_file, 'created')
    if fig is not None:
        plt.close(fig)
