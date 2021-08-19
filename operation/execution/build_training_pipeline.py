# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse

from azureml.core import Datastore, Environment
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration

from utils import config, workspace, compute, pipeline


def main(dataset_name, model_name, pipeline_name, compute_name, environment_path,
         model_metric_name, maximize, keep_columns, target_column, target_values, 
         text_columns, fold_number, training_script_name, pipeline_version=None):

    # Retrieve workspace
    ws = workspace.retrieve_workspace()

    # Training setup
    compute_target = compute.get_compute_target(ws, compute_name)
    env = Environment.load_from_directory(path=environment_path)
    run_config = RunConfiguration()
    run_config.environment = env

    # Create a PipelineData to pass data between steps
    pipeline_data = PipelineData(
        'pipeline_data', datastore=Datastore.get(ws, os.getenv('DATASTORE_NAME', 'workspaceblobstore'))
    )

    # Create steps
    train_step = PythonScriptStep(
        name="Train Model",
        source_directory="src",
        script_name=training_script_name,
        compute_target=compute_target,
        outputs=[pipeline_data],
        arguments=[
            '--dataset-name', dataset_name,
            '--model-name', model_name,
            '--output-dir', pipeline_data,
            '--model-metric-name', model_metric_name,
            '--keep-columns', keep_columns,
            '--target-column', target_column,
            '--target-values', target_values,
            '--text-columns', text_columns,
            '--fold-number', fold_number,
        ],
        runconfig=run_config,
        allow_reuse=False
    )

    evaluate_step = PythonScriptStep(
        name="Evaluate Model",
        source_directory="src",
        script_name="evaluate.py",
        compute_target=compute_target,
        inputs=[pipeline_data],
        arguments=[
            '--model-dir', pipeline_data,
            '--model-name', model_name,
            '--model-metric-name', model_metric_name,
            '--maximize', maximize
        ],
        runconfig=run_config,
        allow_reuse=False
    )

    register_step = PythonScriptStep(
        name="Register Model",
        source_directory="src",
        script_name="register.py",
        compute_target=compute_target,
        inputs=[pipeline_data],
        arguments=[
            '--model-dir', pipeline_data,
            '--model-name', model_name
        ],
        runconfig=run_config,
        allow_reuse=False
    )

    # Set the sequence of steps in a pipeline
    evaluate_step.run_after(train_step)
    register_step.run_after(evaluate_step)

    # Publish training pipeline
    published_pipeline = pipeline.publish_pipeline(
        ws,
        name=pipeline_name,
        steps=[train_step, evaluate_step, register_step],
        description="Model training/retraining pipeline",
        version=pipeline_version
    )

    print(f"Published pipeline {published_pipeline.name} version {published_pipeline.version}")


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == "__main__":
    args = parse_args()

    # Get argurments from environment (these variables are defined in the yml file)
    main(
        model_name=config.get_env_var("AML_MODEL_NAME"),
        dataset_name=config.get_env_var("AML_DATASET"),
        pipeline_name=config.get_env_var("AML_TRAINING_PIPELINE"),
        compute_name=config.get_env_var("AML_TRAINING_COMPUTE"),
        environment_path=config.get_env_var("AML_TRAINING_ENV_PATH"),
        model_metric_name=config.get_env_var("TRAINING_MODEL_METRIC_NAME"),
        maximize=config.get_env_var("TRAINING_MAXIMIZE"),
        keep_columns=config.get_env_var("KEEP_COLUMNS"),
        target_column=config.get_env_var("TARGET_COLUMN"),
        target_values=config.get_env_var("TARGET_VALUES"),
        text_columns=config.get_env_var("TEXT_COLUMNS"),
        fold_number=config.get_env_var("FOLD_NUMBER"),
        training_script_name =config.get_env_var("TRAINING_SCRIPT_NAME"),
        pipeline_version=args.version
    )
