# About this repository

This repository contains the basic repository structure for delivering classification solutions for machine learning (ML) projects based on Azure technologies (Azure ML and Azure DevOps). 

The folder names and files are chosen based on personal experience. You can find the principles and ideas behind the structure, which we recommend to follow when customizing your own classification project and MLOps process. Also, we expect users to be familiar with AML concepts and how to use the Azure technology.

# Prerequisites

In order to successfully complete your solution, you will need to have access to and or provisioned the following:

- Access to an Azure subscription
- Access to an Azure Devops subscription
- Service Principal

# Getting Started

If you want to deploy one of the use cases provided in the forks, have a look at how to setup your environment in your own Azure DevOps Organisation in  **[end-to-end setup documentation](./docs/how-to/EndToEndSetup.md)**.

The steps you will need to follow are:

1. Setting up the Azure infrastructure

   - For general best-practices, we invite you to visit the official [Cloud Adoption Framework](https://docs.microsoft.com/en-us/azure/cloud-adoption-framework/ready/azure-best-practices/ai-machine-learning-resource-organization?branch=pr-en-us-1541) 

   - if you are starting with MLOps, you will find the necessary Azure Devops pipelines and ARM templates in the folder _infrastructure_ to setup the recommended infrastructure.

   - if you already have a preferred architecture and Azure resources, you can delete the infrastructure folder. In the [end-to-end setup documentation](./docs/how-to/EndToEndSetup.md), you will find a FAQ section with steps to follow to adapt the code to your infrastructure.

2. Creating your registered dataset to Azure Machine Learning studio. In the folder **./docs/data** you will find the sample CSV file and the instruction file to setup your registered dataset in Azure Machine Learning (AML) studio.

3. Creating your CI/CD Pipelines (one for model training and the other for batch inference) to Azure Devops. In the folder **./azure-pipelines** you will find the yaml file to setup your CI/CD pipeline in Azure Devops (ADO).

If you have managed to run the entire example, well done! You can now adapt the same code to your own use case with the exact same infrastructure and CI/CD pipeline. To do so, follow these steps:

- Add your variables (model and dataset name, azure environment, ...) in [configuration-aml.variables.yaml](./configuration/configuration-aml.variables.yml) in the _configuration folder_

- Add your core ML code (feature engineering, training, scoring, etc) in **./src**. We provide the structure of the core scripts. You can fill the core scripts with your own functions. For examle, if your data scientist has already finished model training task and picked a best algorithm or classifier, you can use "train1classifier.py" to train your model against the input data.  Otherwise, you can use "train.py" to train your model with multiple classifiers (it can take up to 2-3 hours).  The code will pick up the best performing classifier automatically based on a model metric name which you can specify in the configuration file.

- Add your operation scripts that handle the core scripts (e.g sending the training script to a compute target, registering a model, creating an azure ml pipeline,etc) to **operation/execution**. We provide some examples to easily setup your experiments and AML Pipeline

The project folders are structured in a way to rapidly move from a notebook experimentation to refactored code ready for deployment as following: ![design folder](docs/media/folder_design.PNG =500x)

# General Coding Guidelines

For more details on the coding guidelines and explanation on the folder structure, please go to [data/docs/how-to](docs/how-to/TemplateDocumentation.md).

1. Core scripts should receive parameters/config variables only via code arguments and must not contain any hardcoded variables in the code (like dataset names, model names, input/output path, ...). If you want to provide constant variables in those scripts, write default values in the argument parser.

2. Variable values must be stored in **_configuration/configuration-aml.yml_**. These files will be used by the execution scripts (azureml python sdk or azure-cli) to extract the variables and run the core scripts.

3. Two distinct configuration files for environment creation:
   - (A) for local dev/experimentation: may be stored in the project root folder (requirement.txt or environment.yml). It is required to install the project environment on a different laptop, devops agent, etc.
   - (B) for remote compute: stored in **_configuration/environments/environment_x_** contains only the necessary Python packages and versions if speficied to be installed on remote compute targets or AKS.  One subfolder is for model training and the other is for batch inference.

4. There are only 2 core secrets to handle: the azureml workspace authentication key and a service principal. Depending on your use-case or constraints, these secrets may be required in the core scripts or execution scripts. We provide the logic to retrieve them in a **_utils.py_** file in **_src_** and a **_workspace.py_** in **_operation/execution/utils_**. If you want to run the scripts locally, you need to download a config file from your AML workspace and then copy this config file into your project folder. 

# Default Directory Structure

```
├───azure-pipelines     # folder containing all the azure devops pipelines
│───configuration   # any configuration files
│   ├──compute
│   └──environments
├── docs
│   ├── code            # documenting everything in the code directory (could be sphinx project for example)
│   ├── data            # documenting datasets, data profiles, behaviors, column definitions, etc
│   ├── how-to          # documents on how to use this template and how to setup the environment.
│   ├── media           # storing images, videos, etc, needed for docs.
│   └── references      # for collecting and documenting external resources relevant to the project.
├───notebooks           # experimentation folder with notebooks, code and other. The files don't need to be committed
├───operation           # all the code and configuration to execute the source scripts
│   ├───execution       # azure ml scripts to run source script on remote
│   ├───monitoring      # anything related to monitoring, model performance, data drifts, model scoring, etc
│   └───tests           # for testing your code, data, and outputs
│       ├───data_validation     # any data validation scripts
│       └───unit                # unit tests
|── src
├── .gitignore
├── README.md
└── requirements.txt
```

# FAQ
Frequently asked questions (FAQ) can be found in [how-to](docs/how-to/FAQ.md)
# LICENCE

Copyright (c) Microsoft Corporation

All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ""Software""), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE