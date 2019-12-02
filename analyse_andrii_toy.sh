#!/usr/bin/env bash

folders=$(ls -d simulations/*/*)
model='C16x16_U2_C32x16_U2_C64x8_U2_C128x8_C64x4_C32x4_C16x2_C4x2_C1x1_C1x1_ns0.1_shift64_all1-50-10lr0.0002smooth1_amsgrad_run0'

for folder in ${folders}; do
    files_pix1=$(ls ${folder}/*_Ch1.root)
    for file_pix1 in ${files_pix1}; do
        file_pix2=${file_pix1/Ch1.root/Ch2.root}
        if [ ! -r "${file_pix2}" ]; then
            echo "missing datafile ${file_pix2}"
            continue
        fi
        output_image=${file_pix1/Ch1.root/g2.png}
        if [ -f ${output_image} ]; then
            echo "skip ${file_pix1} analysis as ${output_image} exists"
            continue
        fi
        echo "analyse ${file_pix1}"
        python analyse_andrii_toy.py ${model} ${file_pix1} ${file_pix2} ${output_image}
    done
done