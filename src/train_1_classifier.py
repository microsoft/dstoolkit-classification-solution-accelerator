# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support

from azureml.core import Run, Dataset, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.run import _OfflineRun

from utils import retrieve_workspace, get_dataset
from ml_utils import preprocessing

def main(dataset_name, model_name, output_dir, model_metric_name, keep_columns, target_column, target_values, text_columns, fold_number):

    run = Run.get_context()
    ws = retrieve_workspace()

    print("Getting dataset...")
    data = get_dataset(ws, filename=dataset_name)

    print("Preprocessing data...")
    data = preprocessing(data, keep_columns, target_column, target_values)

    print("Splitting data into a training and a testing set...")
    X_train, X_test, y_train, y_test = train_test_split_randomly(data, target_column)

    print("Training model...")
    model = train(X_train, y_train, text_columns)

    print("Evaluating model...")
    metrics = get_model_metrics(model, X_test, y_test, model_metric_name)

    print("Save metrics in run...")
    if not isinstance(run, _OfflineRun):
        for k, v in metrics.items():
            run.log(k, v)
            if run.parent is not None:
                run.parent.log(k, v)

    print(f"Saving model in folder {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)

    print('Finished.')


def train_test_split_randomly(df, target_column):
    """
    Split dataframe into random train and test subset
    """
    y = df[target_column]
    X = df.drop(target_column, axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, text_columns):
    """
    Training function
    """

    print("Start training")

    text_features = text_columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).drop(text_features, axis=1).columns

    # There are the three feature extraction pipelines for numeric, categorical, and text features respectively
    #  The documentation on these preprocessings can be found at
    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
    # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer())])

    transformer = [ ('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)]
    for feature in text_features:
        t = [('txt', text_transformer, feature)]
        transformer = t + transformer

    # Column transformer that packaged does the feature extractions defined above to the specified lists of features
    preprocessor = ColumnTransformer(
        transformers = transformer)
 
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier())])
    model = clf.fit(X_train, y_train)
    return model


def get_model_metrics(model, X_test, y_test, model_metric_name):
    """"
    Get model metrics
    """
    metrics = {}
    pred = model.predict(X_test)

    scores = precision_recall_fscore_support(y_test, pred, average='micro')
    metrics['Precision'] = scores[0]
    metrics['Recall'] = scores[1]
    metrics['F1_score'] = scores[2]

    return metrics


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='two_class2')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--model-name', type=str, default='two_class.pkl')
    parser.add_argument('--model-metric-name', type=str, default='mse',
                        help='The name of the evaluation metric used in Train step')
    parser.add_argument('--keep-columns', type=str, default='Helpfulness Score|Score|Text|Target')
    parser.add_argument('--target-column', type=str, default='Target')
    parser.add_argument('--target-values', type=str, default='toys games|not a toy/game')
    parser.add_argument('--text-columns', type=str, default='Text')
    parser.add_argument('--fold-number', type=int, default=4)

    args_parsed = parser.parse_args(args_list)

    return args_parsed


if __name__ == '__main__':
    args = parse_args()

    main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        model_metric_name=args.model_metric_name,
        keep_columns=args.keep_columns.split('|'),
        target_column=args.target_column,
        target_values=args.target_values.split('|'),
        text_columns=args.text_columns.split('|'),
        fold_number=args.fold_number,        
    )
