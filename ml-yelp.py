#!/usr/bin/env python

import pandas as pd
import os, sys
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder


def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

def main():
    names = ['address', 'attributes',
            'city', 'hours', 'latitude',
            'longitude', 'neighborhood', 'postal_code',
            'review_count', 'stars', 'state','type',
            'is_open']

    categorical_cols = ['address', 'attributes', 'city', 'hours',
                        'neighborhood', 'postal_code', 'state', 'type']
    
    cwd = os.getcwd()
    df = pd.read_json(path_or_buf=cwd+"/data/valid_json.json")
    onehotdf = one_hot(df, categorical_cols) 
    new_df = onehotdf.filter(names, axis=1)

    print(new_df.head(1))
    array = new_df.values
    X = array[:,0:12]
    y = array[:,12]

    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X,y)

    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)

    print(features.head(2))

if __name__ == "__main__":
    main()
