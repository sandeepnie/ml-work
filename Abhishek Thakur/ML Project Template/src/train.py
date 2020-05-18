# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:57:05 2020

@author: Sandeep
"""
import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

import joblib

from . import dispatcher

# TRAINING_DATA = os.environ.get("input/train_folds.csv")
# TEST_DATA = os.environ.get("TEST_DATA")
# FOLD = os.environ.get("FOLD")
# MODEL = os.environ.get("MODEL")

TRAINING_DATA =  "input/train_folds.csv"
TEST_DATA = "input/test.csv"
FOLD = 0
# MODEL = os.environ.get("MODEL")
FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}


if __name__ == "__main__":
    

    
    df = pd.read_csv(TRAINING_DATA)
    FOLD =0
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]
    
    ytrain = train_df.target.values
    yvalid = valid_df.target.values
    
    train_df = train_df.drop(["id","target","kfold"], axis =1)
    valid_df = valid_df.drop(["id","target","kfold"], axis =1)
    
    valid_df = valid_df[train_df.columns]
    
    #label encoder
    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c, lbl))
        
    #data is ready to train
    clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2)
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(preds)
    print(metrics.roc_auc_score(yvalid,preds))
    
    #save the model
    joblib.dump(label_encoders,f"models/randomforest_label_encoder.pkl")
    joblib.dump(clf,f"models/randomforest.pkl")    
    
    