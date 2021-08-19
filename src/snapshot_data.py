# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azureml.core import Datastore
from azureml.opendatasets import OjSalesSimulated

from utils import retrieve_workspace

def main(datastore, data_path, maxfiles=None):

    # Pull the data
    oj_sales_files = OjSalesSimulated.get_file_dataset()
    if maxfiles:
        oj_sales_files = oj_sales_files.take(maxfiles)

    # Download the data
    file_paths = oj_sales_files.download(data_path, overwrite=True)

    # Save data snapshot in datastore
    ws = retrieve_workspace()
    datastore = Datastore(ws, name=datastore)
    datastore.upload(
        src_dir=data_path,
        target_path=data_path,
        overwrite=False
    )

    print(f'{len(file_paths)} files downloaded and saved in datastore {datastore}, path {data_path}')


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datastore', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--maxfiles', type=int, default=10)
    args_parsed = parser.parse_args(args_list)

    args_parsed.maxfiles = None if args_parsed.maxfiles <= 0 else args_parsed.maxfiles

    return args_parsed


if __name__ == "__main__":
    args = parse_args()

    main(
        datastore=args.datastore,
        data_path=args.path,
        maxfiles=args.maxfiles
    )
