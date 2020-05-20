# -*- coding: utf-8 -*-


from sklearn import ensemble

TRAINING_DATA = "input/train_folds.csv"
TEST_DATA = "input/test.csv"
FOLD = 0

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}

