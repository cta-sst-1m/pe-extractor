#!/usr/bin/env python3
"""
analyze interferometry data

Usage:
  analyze_data [options] [--] <INPUT>

Options:
  -h --help                   Show this screen.
  --file_off=FILE             File with data either hv off or in dark (for baseline determination).
  --max_events=N              Maximum number of events to analyze. If none,
                              the files are fully read. [Default: None]
  --output_file=OUTPUT        File where to store the results (arrays with one value for each
                              shift_in_sample):
                              * sum1_wf, sum2_wf: sum of waveform of pixels 1 and 2
                              * sum12_wf: sum of waveform of pixels 1 * waveform of pixels 2
                              * n_sample_wf: number of samples analyzed
                              * n_sample_pb, sum12_pb, sum1_pb, sum2_pb: if a model is sepecified,
                              it will be used to compute photo-electron probabilities. The same
                              quantities than for the waveforms are then computed using those
                              probailities instead.
                              [default: ./g2.npz]
  --model=NAME                name of the model to use. If none, no pe 
                              detection is done. [Default: None]
  --shift_in_sample=INTLIST   Comma-separated list of integers. Each represent 
                              a shift in number of samples that correlation are 
                              calculated with.
"""
from docopt import docopt
from pe_extractor.intensity_interferometry import plot_g2_exp
import numpy as np


def convert_list_int(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        text = text.split(',')
        list_int = list(map(int, text))
        return np.array(list_int)


def convert_int(text):
    if text is None or text.lower() == 'none':
        return None
    else:
        return int(text)


if __name__ == '__main__':
    args = docopt(__doc__)
    file_data = args['<INPUT>']
    file_off = args['--file_off']
    max_events = convert_int(args['--max_events'])
    output_file = args['--output_file']
    model = args['--model']
    shift_in_sample = convert_list_int(args['--shift_in_sample'])
    plot_g2_exp(
        file_data, file_off, run_name=model, shift_in_sample=shift_in_sample,
        g2_file=output_file
    )
