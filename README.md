# pe-extractor
Author: Y. Renier
Software to reconstruct photo electrons arrival time from the waveforms using machine learning. 

# Installation
You need tensorflow and keras to use pe-extractor.

# Usage
- look at train_cnn() function in train_cnn.py to see an example on how to train the neural network.
- look at continue_train_cnn() function in train_cnn.py to see an example on how to load an existing neural network and resume its trainning.
- look at run_prediction() in plot_cnn.py to see an example on how to load an existing neural network and to prediction.
- look at the plot_??? and histo_??? functions in plot_cnn.py  to see an example on how to use the output of the neural network.
A very simplistic toy Monte-Carlo simulation of photon arrival time and corresponding waveform is included in toy.py. 
The toy is using pure Poisson statistics with Gaussian noise on top: no baseline, no cross-talk, no amplitude jitter, no after-pulse are taken into account.
