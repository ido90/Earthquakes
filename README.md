# Earthquakes
Research for Kaggle's challenge of earthquakes prediction.

## Contents

<!--ts-->

[Introduction](#introduction)

[Data](#data)

[Representation](#representation)

[Features](#features)

<!--te-->

# Introduction

## The competition

## This repo

# Data

## Signals
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

## Distributions
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

## Fourier transform with missing points

The general solution to missing data points in spectral analysis is apparently to *fill in the missing points using some interpolation*, e.g. linear, Gaussian filter, cubic spline.

Gaussian filter seems more appropriate for general cases of non-uniform samples, where we want to change the whole sampling grid. For just some missing points I believe that *local interpolation (linear/spline) would be better*.

More advanced, barely mentioned solution may be some kind of Fourier interpolation.

All references look quite consistent.
The first one demonstrates the effects on the spectrum.
The third one also suggests doing FFT for each clean interval separately, then averaging them.

http://mres.uni-potsdam.de/index.php/2017/08/22/data-voids-and-spectral-analysis-dont-be-afraid-of-gaps/

https://scicomp.stackexchange.com/questions/593/how-do-i-take-the-fft-of-unevenly-spaced-data

https://dsp.stackexchange.com/questions/22930/spectral-analysis-of-a-time-series-with-missing-data-points


Implementation of interpolation (of y vs. t):
np.interp(full_t, partial_t, partial_y) # linear
sp.interpolate.interp1d(partial_t, partial_y, kind=kind)(full_t) # kind in ‘slinear’, ‘quadratic’, ‘cubic’ yields a spline.
df.interpolate() # method supports most of scipy's interp1d kinds.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html

# Features

## Wavelets
I remembered my MSc supervisor mentioning working on some basis of wavelets which allows controlled trade-off between the locality of the standard basis and the frequency-resolution of the Fourier basis.

Apparently this is a quite known concept in wavelets, and my supervisor's contribution was mainly in high-dimensional data (*Laplacian multiwavelets bases for high-dimensional data* [[1](https://www.google.com/url?q=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS1063520314000918&sa=D&usd=2&usg=AFQjCNGbyxScv1Zy46ae9FGYaMbVYpoMug),[2](https://www.google.com/url?q=http%3A%2F%2Fweb.math.princeton.edu%2F~nsharon%2FNir_Sharon_page%2FLMW.html&sa=D&usd=2&usg=AFQjCNGBs03Qq63koJrOeEfmDn2H6MaLHg)]), which is less relevant for the 1D signal in the current challenge.
I believe the (quite simple) 1D analog is something like [this](https://www.google.com/url?q=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FMexican_hat_wavelet&sa=D&usd=2&usg=AFQjCNEdfbbELq4OGJiRksgDZwbOEMAEjw) or maybe [this](https://www.google.com/url?q=https%3A%2F%2Fjournals.sagepub.com%2Fdoi%2Fabs%2F10.1177%2F1077546317707103%3FjournalCode%3Djvcb&sa=D&usd=2&usg=AFQjCNHUiuEJ70-_RbM8avSPBTFMQpHw-w).
