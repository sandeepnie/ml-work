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
import sys

import joblib


from . import dispatcher

TRAINING_DATA = dispatcher.TRAINING_DATA
TEST_DATA = dispatcher.TEST_DATA
FOLD = dispatcher.FOLD
MODEL = str(sys.argv[1])

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}


if __name__ == "__main__":    
    
    for FOLD in range(5):
        print("Running the FOLD:",FOLD)
        df = pd.read_csv(TRAINING_DATA)
        df_test = pd.read_csv(TEST_DATA)
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df[df.kfold==FOLD].reset_index(drop=True)
        
        ytrain = train_df.target.values
        yvalid = valid_df.target.values
        
        train_df = train_df.drop(["id","target","kfold"], axis =1)
        valid_df = valid_df.drop(["id","target","kfold"], axis =1)
        
        valid_df = valid_df[train_df.columns]
        
        #label encoder
        label_encoders = {}
        for c in train_df.columns:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
            train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
            valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
            label_encoders[c]  = lbl
            
        #data is ready to train
        clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, ytrain)
        preds = clf.predict_proba(valid_df)[:, 1]
        print(preds)
        print(metrics.roc_auc_score(yvalid,preds))
        
        #save the model
        joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
        joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
        joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")    
    
    