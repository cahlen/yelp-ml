#!/usr/bin/env python

import pandas as pd
import os, sys
import numpy as np
from sklearn import preprocessing, svm, cross_validation, tree, metrics
#from sklearn.ensemble import ske
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(0)

def preprocess_data(df, cols):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df[cols] = df[cols].apply(le.fit_transform)
    return processed_df

def main():
    categorical_cols = ['neighborhood','type','state', 'city','name']
    names = ['neighborhood', 'type', 'state', 'city', 'name',
             'stars', 'postal_code', 'latitude', 'longitude', 'review_count', 'is_open']

    # Get json from file into dataframe
    cwd = os.getcwd()
    raw_df = pd.read_json(path_or_buf=cwd+"/data/valid_json.json")

    # Drop rows where zipcode is not an integer
    filtered_df = raw_df[raw_df.postal_code.apply(lambda x: x.isnumeric())]

    # Get sample of rows
    df = filtered_df.filter(names, axis=1) #.sample(n=100000)

    # Convert categorical columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('category'))
    df[['postal_code']] = df[['postal_code']].apply(lambda x: x.astype('int64'))

    df = preprocess_data(df, categorical_cols)

    X = df.drop(['is_open'], axis=1).values
    y = df['is_open'].values

    # Cross validate
    X_train, X_test, y_train, y_test =cross_validation.train_test_split(X,y,test_size=0.3)

    # Decision Tree
    clf_dt = tree.DecisionTreeClassifier(max_depth=10)
    clf_dt.fit(X_train, y_train)
    print("Decision Tree: " + str(clf_dt.score(X_test,y_test)))

    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=50)
    clf_rf.fit(X_train, y_train)
    scores = cross_val_score(clf_rf, X_train, y_train)
    print("Random Forest: " + str(clf_rf.score(X_test,y_test)))
    print("Cross Validation Mean: " + str(scores.mean()))

    # Gradient Boosting Classifier
    clf_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
    clf_gb.fit(X_train, y_train)
    scores = cross_val_score(clf_gb, X_train, y_train)
    print("Gradient Boosting: " + str(clf_gb.score(X_test,y_test)))
    print("Cross Validation Mean: " + str(scores.mean()))


if __name__ == "__main__":
    main()
