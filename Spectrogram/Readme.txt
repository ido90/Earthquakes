Datasets results:
(2) badly-normalized librosa mel-spectrograms.
(3) better-normalized librosa mel-spectrograms.
(4) manual spectrograms.
(5) manual widened spectrograms - intended to be randomly cropped for better sampling.

Notebooks:
- Spectrograms_librosa: display and generate spectrograms using librosa (corresponding to datasets 2,3).
- Spectrograms_librosa_public: edited notebook for publishing as a Kaggle Kernel.
- Spectrograms_manual: display and generate spectrograms manually (corresponding to dataset 4).
- Spectrograms_manual_175K: generate widened manual spectrograms (corresponding to dataset 5, see explanation above).
- NN_spects: train & test CNN on spectrograms.
- NN_test: research some stacking ideas for the models from various cross-validation folds (mainly probabilistic interpretation of classification predictions), and run on test data.
