#!/usr/bin/env bash

model="C16x16_U2_C32x16_U2_C64x8_U2_C128x8_C64x4_C32x4_C16x2_C4x2_C1x1_C1x1_ns0.1_shift64_all1-50-10lr0.0002smooth1_amsgrad_run0"
#max_events=5000
shift_sample_max=25

if [ $# -ge 1 ]; then
    file_in=$1
fi
if [ $# -eq 2 ]; then
    file_off=$2
elif [ $# -eq 1 ]; then
    file_off=$1
else
    echo usage: $0 imput_raw_waveform.root [lampoff_raw_waveform.root]
    exit
fi

shifts_sample=$(echo $(seq -${shift_sample_max} ${shift_sample_max})| tr -s ' ' ',')
if [ -z "$max_events" ]; then
    max_events_opt=""
else
    max_events_opt="--max_events=${max_events}"
fi
shift_opt="--shift_in_sample=${shifts_sample}"
output_file=${file_in%_raw_waveforms.root}_g2.npz
g2_plot=${output_file%.npz}.png
python3 analyze_data.py ${shift_opt} ${max_events_opt} --model=${model} --file_off=${file_off} --g2_plot=${g2_plot} --output_file=${output_file} ${file_in}
