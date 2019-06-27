# pe-extractor
Author: Y. Renier

Software to reconstruct photo-electrons (p.e.) arrival time from the waveforms using machine learning. 

# Installation
You need tensorflow and keras to use pe-extractor.

# Usage
- look at train_cnn() function in train_cnn.py to see an example on how to train the neural network.
- look at continue_train_cnn() function in train_cnn.py to see an example on how to load an existing neural network and resume its trainning.
- look at run_prediction() in plot_cnn.py to see an example on how to load an existing neural network and to prediction.
- look at the plot_??? and histo_??? functions in plot_cnn.py  to see an example on how to use the output of the neural network.

A very simplistic toy Monte-Carlo simulation of photon arrival time and corresponding waveform is included in toy.py. 
The toy is using pure Poisson statistics with Gaussian noise on top: no baseline, no cross-talk, no amplitude jitter, no after-pulse are taken into account.

An example of a pre-trained CNN (cnn-example.h5) is included in the Model folder.
It was designed to use it with data from DigiCam (the camera of the SST-1M telescope).
The default config and the included CNN reflect that:
- 250 MHz sampling rate
- 5 LSB gain (maximum amplitude of 1 p.e.).
- waveforms are 90 samples long

This CNN has been trained with:
- 100 steps per epoch
- 400 waveforms per batch
- 1e-3 of learning rate
- random p.e. rate between 0 and 200 MHz.

It was initially trained with no noise 4 ns smoothing, then 
the smoothing was reduced to 2, and finally the noise had been increase to 2 LSB. 
Those training steps were typically ~100 epochs long.
As the model is light, 100 epochs with 100 batches each with 400 waveforms
took ~20 min to train on a very modest GeForce 940MX (~.25 ms per waveform). 
Training on CPU is about 4x slower.
Prediction is about 20x faster than learning (~0.5 ms per waveform). 

