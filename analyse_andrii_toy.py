import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pe_extractor.cnn import g2_andrii_toy

if len(sys.argv) != 5:
    print(
        sys.argv[0],
        'takes 4 arguments: model datafile_pix1 datafile_pix2 output_image'
    )
    print(len(sys.argv)-1, 'arguments given')
    exit(1)

model = sys.argv[1]
datafile_pix1 = sys.argv[2]
datafile_pix2 = sys.argv[3]
output_image = sys.argv[4]

g2_andrii_toy(
    model, datafile_pix1, datafile_pix2, batch_size=64,
    n_sample=4320, margin_lsb=9, samples_around=5,
    shifts=range(-40, 41), xlim=(-20, 20),
    parallel_iterations=48, plot=output_image
)