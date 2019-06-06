# Summary
This repo contains my humble part in the efforts of Kopeyka team (including Zahar Chikishev in full-time and myself in part-time) in **Kaggle's [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview) competition**.

Our team won a **silver medal**, reaching the **87th place among 4562 competitors** who chased after competition prizes of 50K$.

My main task in the team concentrated on **Neural-Network solutions based directly on raw-signal** (rather than on engineered features).

The Transformer Network in this repo achieved score which would have won a gold medal. Unfortunately, no evidence indicated the quality of this model until the end of the competition, thus we did not submit it as our final solution.

### Competition
Task: given a series of 150K seismic measurements, predict the time remaining until the next earthquake.

Goal function is Mean Absolute Error of predictions.
The train and test sets given in the competition correspond to several thousands of such 150K-long segments.

The data is based on an experiment in which small-scale earthquakes were generated within a laboratory.

### Repo contents

<!--ts-->

- [**Spectrogram-based CNN**](https://github.com/ido90/Earthquakes/tree/master/Spectrogram): [generating](https://www.kaggle.com/idog90/lanl-competition-why-do-spectrograms-fail) spectrogram-images and [applying](https://github.com/ido90/Earthquakes/blob/master/Spectrogram/NN_spects.ipynb) the standard Resnet-34 Convolutional NN on them to predict next quake's time - using both regression and time-intervals-classification (the latter allowing later stacking strategies with probabilistic approach).

- [**Transformer networks**](https://github.com/ido90/Earthquakes/blob/master/Transformer/transformer-network.ipynb): [Attention-based Neural Network](https://arxiv.org/abs/1706.03762), based on [this](https://www.kaggle.com/buchan/transformer-network-with-1d-cnn-feature-extraction) great public Kaggle kernel.

- [**Final model selection**](https://github.com/ido90/Earthquakes/blob/master/Features%20Analysis/final_models_analysis.ipynb): detailed analysis of the final candidate-models for submission.

- [Features EDA](https://github.com/ido90/Earthquakes/tree/master/Features%20Analysis): analysis of the models' predictions and of the features they use.

- Others:
   - [Basic EDA](https://github.com/ido90/Earthquakes/blob/master/Others/README.md#very-basic-explanatory-data-analysis): some very basic figures of the data and its distribution.
   - [Representation](https://github.com/ido90/Earthquakes/blob/master/Others/README.md#representation): basic research of the representation of the signal in Fourier space, with consideration of the signal samples not being exactly unifrom over time (also see [dedicated research of FFT with dropped samples](https://github.com/ido90/SignalReconstruction)).
   - [Features](https://github.com/ido90/Earthquakes/blob/master/Others/README.md#wavelets-features): brief discussion of basic wavelets features.

<!--te-->

### Best Leaderboard score :)
(corresponding to the public score which is not the competition's final score)
![](https://github.com/ido90/Earthquakes/blob/master/Best%20Leaderboard%20Score.png)
