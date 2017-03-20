#!/usr/bin/env python

import pandas as pd
import os, sys
import numpy as np
from sklearn import preprocessing, svm, cross_validation, tree, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

# Random seed for models
np.random.seed(0)

# Global vars
categorical_cols = ['neighborhood', 'type', 'state', 'city', 'name']
names = ['neighborhood', 'type', 'state', 'city', 'name',
        'stars', 'postal_code', 'latitude', 'longitude', 
        'review_count', 'is_open']



def preprocess_data(df, cols):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df[cols] = df[cols].apply(le.fit_transform)
    return processed_df

def custom_processing(df):
    # drop non numeric zip codes
    filtered_df = df[df.postal_code.apply(lambda x: x.isnumeric())]
    named_df = filtered_df.filter(names, axis=1)
    named_df[categorical_cols] = named_df[categorical_cols].apply(lambda x: x.astype('category'))
    named_df[['postal_code']] = named_df[['postal_code']].apply(lambda x: x.astype('int64'))
    return named_df



def load_data():
    cwd = os.getcwd()
    return pd.read_json(path_or_buf=cwd+"/data/valid_json.json")

def main():
    raw_df = load_data()
    df = custom_processing(raw_df)
    df = preprocess_data(df, categorical_cols)

    X = df.drop(['is_open'], axis=1).values
    y = df['is_open'].values

    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    # Feature Selection
    # - Change k to number of k best features you want to keep
    chi2k = SelectKBest(chi2, k=2) 
    X_new_train = chi2k.fit_transform(X_scaled, y_scaled)
    mask = chi2k.get_support()
    new_features = []
    for bool, feature in zip(mask, names):
        if bool:
            new_features.append(feature)

    # Create new dataframe from select features
    # and scale features again.
    X_pre_split = scaler.fit_transform(df[new_features].values)
    y_pre_split = scaler.fit_transform(df['is_open'].values)

    # Split data 
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_pre_split,
            y_pre_split,test_size=0.3)

    # Decision Tree
    clf_dt = tree.DecisionTreeClassifier(max_depth=10)
    clf_dt.fit(X_train, y_train)
    scores = cross_val_score(clf_dt, X_train, y_train)
    print("Decision Tree: " + str(clf_dt.score(X_test ,y_test)))
    print("Cross Validation Mean: " + str(scores.mean()) + "\n")

    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=50)
    clf_rf.fit(X_train, y_train)
    scores = cross_val_score(clf_rf, X_train, y_train)
    print("Random Forest: " + str(clf_rf.score(X_test,y_test)))
    print("Cross Validation Mean: " + str(scores.mean()) + "\n")

    # Gradient Boosting Classifier
    clf_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
    clf_gb.fit(X_train, y_train)
    scores = cross_val_score(clf_gb, X_train, y_train)
    print("Gradient Boosting: " + str(clf_gb.score(X_test,y_test)))
    print("Cross Validation Mean: " + str(scores.mean()) + "\n")

if __name__ == "__main__":
    main()
