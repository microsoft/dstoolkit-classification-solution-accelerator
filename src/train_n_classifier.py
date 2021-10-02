# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Import all needed Python packages and functions
from azureml.core import Run, Dataset, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.run import _OfflineRun

import os
import argparse

import joblib
import pandas as pd
import numpy as np
from numpy.random import uniform

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import xgboost
from xgboost import XGBClassifier

import statistics as stats


from utils import retrieve_workspace, get_dataset
from ml_utils import preprocessing

def main(dataset_name, model_name, output_dir, model_metric_name, keep_columns, target_column, target_values, text_columns, fold_number):
    run = Run.get_context()
    ws = retrieve_workspace()

    print("Getting dataset...")
    data = get_dataset(ws, filename=dataset_name)

    data = preprocessing(data, keep_columns, target_column, target_values)

    print("Training model...")
    model, metrics = train(data, model_metric_name, target_column, text_columns, fold_number)

    print("Saving metrics in run...")
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

def train(df, model_metric_name, target_column, text_columns, fold_number):
    """
    Training function
    """
    # Do your training here
    print("Start training") 
    # Create list of text and numeric and categorical features
    y = df[target_column]
    X = df.drop(target_column, axis = 1)
    
    text_features = text_columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).drop(text_features, axis=1).columns

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

    # Column transformer does the feature extractions defined above to the specified lists of features
    preprocessor = ColumnTransformer(
        transformers = transformer, remainder="drop")

    ###This is the list of classifiers that we will train and test
    #  But remove those which do NOT have forecasting probabilities
    classifiers = [
        #Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),
      
        #GLM
        linear_model.LogisticRegressionCV(),
        #linear_model.PassiveAggressiveClassifier(),
        #linear_model.RidgeClassifierCV(),
        #linear_model.SGDClassifier(),
        #linear_model.Perceptron(),
        
        #Navies Bayes
        naive_bayes.BernoulliNB(),

        #Nearest Neighbor
        neighbors.KNeighborsClassifier(),
        
        #SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        #svm.LinearSVC(),
        
        #Trees    
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),
        
        #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        XGBClassifier()    
        ]

    # Creating a dataframe to store for model metrics each model, 
    #  'Model' itself and well as the macro scores from n (defined above) cross validations
    MLA_columns = ['Classifier Name', 'Model', 'Precision', 'Precision STD', 'Recall', 'Recall STD', 'F1', 'F1 STD']

    MLA_compare = pd.DataFrame()
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    n = int(fold_number)

    # Loop through all the classifers to fit them, get their scores, and pick up the best model
    row_index = 0
    for alg in classifiers:
        pipe = Pipeline(steps=[('preprocessor', preprocessor),('classifier', alg)])
        scores = cross_validate(pipe, X, y, scoring = scoring, cv = n)
        df = pd.DataFrame(list(scores.items()),columns = ['score','scores'])

        MLA_name = alg.__class__.__name__
      
        MLA_compare.loc[row_index, 'Classifier Name'] = MLA_name
        MLA_compare.loc[row_index, 'Model'] =  pipe.fit(X, y)
        MLA_compare.loc[row_index, 'Precision'] = df.iloc[2,1].mean()
        MLA_compare.loc[row_index, 'Precision STD'] =  stats.stdev(df.iloc[2,1])
        MLA_compare.loc[row_index, 'Recall'] = df.iloc[3,1].mean()
        MLA_compare.loc[row_index, 'Recall STD'] = stats.stdev(df.iloc[3,1])
        MLA_compare.loc[row_index, 'F1'] = df.iloc[4,1].mean()
        MLA_compare.loc[row_index, 'F1 STD'] = stats.stdev(df.iloc[4,1])
         
        row_index+=1

    # Pick up the best performing model based on the model metric name
    MLA = MLA_compare.sort_values(by = model_metric_name, ascending = False)
    MLA.reset_index(drop = True, inplace = True)  
    model = MLA.loc[0,'Model']
    metrics = MLA.iloc[0][['Precision', 'Recall', 'F1', 'Precision STD']]

    return model, metrics


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='two_class2')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--model-name', type=str, default='two_class.pkl')
    parser.add_argument('--model-metric-name', type=str, default='Recall',
                        help='The name of the evaluation metric used in Train step')
    parser.add_argument('--keep-columns', type=str, default='Helpfulness_Score|Score|Text|Target')
    parser.add_argument('--target-column', type=str, default='Target')
    parser.add_argument('--target-values', type=str, default='toys games|not a toy/game')
    parser.add_argument('--text-columns', type=str, default='Text')
    parser.add_argument('--fold-number', type=str, default='4')

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
