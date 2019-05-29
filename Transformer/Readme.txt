Source:
https://www.kaggle.com/buchan/transformer-network-with-1d-cnn-feature-extraction

Notebooks:
transformer-network: main notebook.
transformer-network_max_train: train one model over all data.

others:
Some variations of the original kernel, that were intended to diagnose the source of different between my variant of transformer and the original one.
Main difference was found to be me sampling training segments with constant alignment of 25K, which turned out to effectively reduce the data and caused overfitting.

folds: table of segments and their folds in CV.
train.meta.covs.pickle: a file of Zahar, used to generate folds for CV.

trained_model_fold{i}.h5: model trained without fold i.
trained_model_full.h5: model trained over all available data.
transformer_predictions.pkl: list of predictions per fold and per model (each model trained w/o one fold).
submission_transformer.csv: competition's submission file based on the transformer models.
