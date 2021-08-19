# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse

import joblib
import pandas as pd
from azureml.core import Dataset, Model

from utils import retrieve_workspace
from ml_utils import preprocessing, predict


def main(dataset_name, model_name, output_dir, output_file, target_column):
    ws = retrieve_workspace()

    print("Getting data for inference...")
    dataset = Dataset.get_by_name(ws, dataset_name)
    data = dataset.to_pandas_dataframe()

    print("Getting model for inference...")
    try:
        model_path = Model.get_model_path(model_name=model_name)
    except Exception:
        print('Model not found in cache. Trying to download locally')
        model_container = Model(ws, name=model_name)
        model_path = model_container.download()

    print("Loading model...")
    with open(model_path, 'rb') as f:
        model = joblib.load(f)

    print("Generating predictions data...")
    data = data.drop(target_column, axis = 1)
    data = predict(model, data)

    print(f"Saving predictions in folder {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    file_forecast = os.path.join(output_dir, output_file)
    data.to_csv(file_forecast, index=False)

    print("Finished.")


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='two_class2')
    parser.add_argument('--model-name', type=str, default='two_class.pkl')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--output-file', type=str, default='predictions.csv')
    parser.add_argument('--target-column', type=str, default='Target')

    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == '__main__':
    args = parse_args()

    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        output_file=args.output_file,
        target_column=args.target_column,
    )
