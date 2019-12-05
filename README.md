# pe-extractor
Author: Y. Renier

Software to reconstruct photo-electrons (p.e.) arrival time from the waveforms using machine learning. 

# Installation
You need tensorflow, keras, tqdm and matplotlib to use pe-extractor.

$ pip install tensorflow keras tqdm matplotlib

Download pe-extractor and install it

$ git clone https://github.com/cta-sst-1m/pe-extractor.git

$ pip install -e .

An optional converter for zfits files is available (zfits_to_raw_wf), to compile it:
$ make

# Usage
Bash scripts are available to compute g² from root files:
- analyze_data.sh to read ROOT files containing a Ttree "waveforms" with the leaves "wf1" and "wf2" containing the synchronized waveforms for the 2 pixels 
- analyse_andrii_toy.sh to read ROOT files from the toy Monte-Carlo . The ROOT files must contain a TTree "A" with the leaf "Npe" containing the true number of photoelectron and the leaf "WaveformAmpliDetected" containing the simulated waveforms for a pixel.

# Quick description
The Extractor class (declared in cnn.py) is used to estimate photo-electron probabilities using waveforms. 

The Keras model used underneath ( in Extractor.model ) can be trained with the Extractor.train() function, or it can be loaded from h5 files with the Extractor.load() function.

The training is done using online a very simplistic toy Monte-Carlo (so no overtraining is possible). 
That toy is using pure Poisson statistics with Gaussian noise on top: no cross-talk, no amplitude jitter, no after-pulse are taken into account.

An example of a pre-trained CNN is included in the Model folder.
