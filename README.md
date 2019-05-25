# Summary
This repo contains my humble part in the efforts of Kopeyka team (including Zahar Chikishev in full-time and myself in part-time) in Kaggle's [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview) competition.

My main task in the team concentrated on NN-based solutions, which did not prove very useful for the competition.

While my work is surely far away from optimal, it is suspected that Neural Networks have lower success potential for the task (in particular, most public discussion in the competition concentrated on trees-based solutions and dedicated feature-engineering).

### Competition
Task: given a series of 150K seismic measurements, predict the time remaining until the next earthquake.

Goal function is Mean Absolute Error of predictions.
The train and test sets given in the competition correspond to several thousands of such 150K-long segments.

The data is based on an experiment in which small-scale earthquakes were generated within a laboratory.

### Repo contents

<!--ts-->

- **Spectrogram-based CNN**: [Generating](https://www.kaggle.com/idog90/lanl-competition-why-do-spectrograms-fail) spectrogram-images and [applying](https://github.com/ido90/Earthquakes/blob/master/Spectrogram/NN_spects.ipynb) the standard Resnet-34 Convolutional NN on them to predict next quake's time - using both regression and time-intervals-classification (the latter allowing later stacking strategies with probabilistic approach).

- [**Transformer networks**](https://github.com/ido90/Earthquakes/blob/master/Transformer/transformer-simplified.ipynb): [Attention-based Neural Network](https://arxiv.org/abs/1706.03762), based on [this](https://www.kaggle.com/buchan/transformer-network-with-1d-cnn-feature-extraction) great public Kaggle kernel.

- [Basic EDA](#very-basic-explanatory-data-analysis): some very basic figures of the data and its distribution.

- [Representation](#representation): basic research of the representation of the signal in Fourier space, with consideration of the signal samples not being exactly unifrom over time (also see [dedicated research of FFT with dropped samples](https://github.com/ido90/SignalReconstruction)).

- [Features](#wavelets-features): brief discussion of basic wavelets features.

<!--te-->


# Very Basic Explanatory Data Analysis

### Signals
It is quite a problem to load all the data naively using read_csv.
Until this issue is handled, only ~10% of the data are actually used (~100M/1B samples).

|![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/train_100M_samples_low_resolution_interactive.png)|
|:--:|
| Train set: first 100M samples (~10%) with skips (only 1 row in a 1000 is shown) |

|![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/train_10M_samples_interactive.png)|
|:--:|
| Train set: first 10M samples (~1%) with no skips |

|![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/train_150K_samples_interactive.png)|
|:--:|
| Train set: a random segment of 150K samples |

|![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/train_150K_samples_interactive_zoom_calm.png)|
|:--:|
|![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/train_150K_samples_interactive_zoom_calm.png)|
| Train set: zoom in |
| Note: it does not seem trivial whether or not there is discontinuity in the transition between the 4096-sized chunks. |

|![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/test_10_segments.png)|
|:--:|
| Test set: 10 random segments (with adjacent indices) of 150K samples each |

|![](https://github.com/ido90/Earthquakes/blob/master/Output/FFT/train_vs_test.png)|
|:--:|
| FFT of samples of train set vs. test set |

### Distributions
Below are plotted the distributions of the complete train and test signals with both linear and log (with base 10) scales.
The whole run required ~10 minutes.

It looks like the orders of magnitudes are similar over the distributions.
Both distributions are within the range (-1)-(+9) for 80% of the time, yet reach +-6000 in their extremes (in which the test set reaches ~20% larger absolute values).
The train signal is larger by 1 in median and by 0.4 in average.

![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/quantile_plots.png)
![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/quantile_plots_log.png)

![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/qqplot.png)
![](https://github.com/ido90/Earthquakes/blob/master/Output/Signal%20description/qqplot_log.png)


# Representation

### Time issues
The train set consists of samples which include time_to_quake entry.
Adjacent times have typical gap of 1ns, even though the measurement device is supposed to have frequency of 1/250ns.
In addition, there are larger gaps every ~4096 samples.

It was speculated that the reported times are the times of the device logger every 4096 samples, and that the correct times are more uniformly spreaded over the large gaps.

To fix that, the large gaps were detected, and the corresponding sequence of times was redefined with gaps of 250ns backwards from the last entry in the sequence.
The remaining gaps are speculated to have 48 missing samples, and were filled up with nans as needed (more meaningful filling will be done next).

The time correction seems to work not quite as expected.
In particular, the remaining gaps after the time correction either don't exist or are too large (305 samples instead of 48).

Since we don't seem to successfully build a coherent time-grid, we will avoid any interpolations that would have allow us to apply FFT on a full uniform grid (as demonstrated [here](https://github.com/ido90/SignalReconstruction)).
Instead, we will just apply FFT separately to every block of 4096 samples.

|![](https://github.com/ido90/Earthquakes/blob/master/Output/FFT/time_correction.png)|
|:--:|
| Corrected times and the corresponding signal |

### Averaged Fourier transform over blocks

Since the data are given in sequences of 4096 uniform samples (up to few exceptions with 4095 sampels), the FFT can just be applied separately to each sequence, allowing to capture any frequency corresponding to period of 4096 samples or shorter.

|![](https://github.com/ido90/Earthquakes/blob/master/Output/FFT/train_averaged_fft.png)|
|:--:|
| An average of all the ~35 FFTs of length 4096 within a train segment of 150K smaples (up), and two arbitrary particular FFTs (down). |

|![](https://github.com/ido90/Earthquakes/blob/master/Output/FFT/train_blocks_fft_are_similar.png)|
|:--:|
| Distribution of the peaks of the various blocks FFTs: the peak is at a constant frequency somewhere between 200-300 samples (up to reflection of the Fourier transform), in amplitude that varies uniformly within 10-60. |

### Bottom line

As can be see in the spectrograms [notebook](https://github.com/ido90/Earthquakes/blob/master/Spectrogram/Spectrogram_librosa.ipynb), and in spite of the promising results in the [general research of this topic](https://github.com/ido90/SignalReconstruction), the spectral analysis of the signal in the competition does not seem to be highly affected by the sampling time issue. This was also confirmed by Zahar's work with FFT-based features.


# Wavelets Features

I remembered my MSc supervisor mentioning working on some basis of wavelets which allows controlled trade-off between the locality of the standard basis and the frequency-resolution of the Fourier basis.

Apparently this is a quite known concept in wavelets, and my supervisor's contribution was mainly in high-dimensional data (*Laplacian multiwavelets bases for high-dimensional data* [[1](https://www.google.com/url?q=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS1063520314000918&sa=D&usd=2&usg=AFQjCNGbyxScv1Zy46ae9FGYaMbVYpoMug),[2](https://www.google.com/url?q=http%3A%2F%2Fweb.math.princeton.edu%2F~nsharon%2FNir_Sharon_page%2FLMW.html&sa=D&usd=2&usg=AFQjCNGBs03Qq63koJrOeEfmDn2H6MaLHg)]), which is less relevant for the 1D signal in the current challenge.
I believe the (quite simple) 1D analog is something like [this](https://www.google.com/url?q=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FMexican_hat_wavelet&sa=D&usd=2&usg=AFQjCNEdfbbELq4OGJiRksgDZwbOEMAEjw) or maybe [this](https://www.google.com/url?q=https%3A%2F%2Fjournals.sagepub.com%2Fdoi%2Fabs%2F10.1177%2F1077546317707103%3FjournalCode%3Djvcb&sa=D&usd=2&usg=AFQjCNHUiuEJ70-_RbM8avSPBTFMQpHw-w).

Since it doesn't look like this could yield any competitive advantage beyond wavelets features in public kernels of the competition, I did not push forward on this direction.
