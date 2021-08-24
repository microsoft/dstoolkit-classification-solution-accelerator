![design folder](https://github.com/microsoft/dstoolkit-classification-solution-accelerator/blob/main/docs/media/Banner_Classification_SA.png)

About this repository
============================================================================================================================================

This repository contains the basic repository structure for delivering classification solutions for machine learning (ML) projects based on Azure technologies (Azure ML and Azure DevOps).

Binary classification models are perhaps the most common use-case in predictive analytics. The reason is that many key client actions across a wide range of industries are binary in nature, such as:

-   defaulting on a loan,
-   clicking on an ad,
-   terminating a subscription,
-   quality control of products, processes or services,
-   medical testing,and so on.

Because of this broad commonality, this accelerator will prove optimal in many data science-based engagements, streamlining time to value.

The folder names and files are chosen based on personal experience. You can find the principles and ideas behind the structure, which we recommend to follow when customizing your own classification project and MLOps process. Also, we expect users to be familiar with AML concepts and how to use the Azure technology.

Details of the accelerator
============================================================================================================================
-   Leverages the ML Ops accelerator to provide a configurable and re-usable solution accelerator for binary classification cases.
-   Auto selects the best classification algorithm for the dataset based on user defined criteria (parameters).
-   Uses Azure ML and Azure Dev Ops


Prerequisites
============================================================================================================================

In order to successfully complete your solution, you will need to have access to and or provisioned the following:

-   Access to an Azure subscription
-   Access to an Azure Devops subscription
-   Service Principal

Getting Started
================================================================================================================================

If you want to deploy one of the use cases provided in the forks, have a look at how to setup your environment in  **[end-to-end setup documentation](https://github.com/microsoft/dstoolkit-classification-solution-accelerator/blob/main/docs/how-to/EndToEndSetup.md)**.

The steps you will need to follow are:

1.  Setting up the Azure infrastructure

    -   For general best-practices, we invite you to visit the official [Cloud Adoption Framework](https://docs.microsoft.com/en-us/azure/cloud-adoption-framework/ready/azure-best-practices/ai-machine-learning-resource-organization?branch=pr-en-us-1541) 

    -   if you are starting with MLOps, you will find the necessary Azure Devops pipelines and ARM templates in the folder *infrastructure* to setup the recommended infrastructure.

    -   if you already have a preferred architecture and Azure resources, you can delete the infrastructure folder. In the [end-to-end setup documentation](https://github.com/microsoft/dstoolkit-classification-solution-accelerator/blob/main/docs/how-to/EndToEndSetup.md), you will find a FAQ section with steps to follow to adapt the code to your infrastructure.

2.  Creating your registered dataset to Azure Machine Learning studio. In the folder **./docs/data** you will find the sample CSV file and the instruction file to setup your registered dataset in Azure Machine Learning (AML) studio.

3.  Creating your CI/CD Pipelines (one for model training and the other for batch inference) to Azure Devops. In the folder **./azure-pipelines** you will find the yaml file to setup your CI/CD pipeline in Azure Devops (ADO).

If you have managed to run the entire example, well done! You can now adapt the same code to your own use case with the exact same infrastructure and CI/CD pipeline. To do so, follow these steps:

-   Add your variables (model and dataset name, azure environment, ...) in [configuration-aml.variables.yaml](https://github.com/microsoft/dstoolkit-classification-solution-accelerator/blob/main/configuration/configuration-aml.variables.yml) in the *configuration folder*

-   Add your core ML code (feature engineering, training, scoring, etc) in **./src**. We provide the structure of the core scripts. You can fill the core scripts with your own functions. For examle, if your data scientist has already finished model training task and picked a best algorithm or classifier, you can use "train_1_classifier.py" to train your model against the input data. Otherwise, you can use "train_n_classifier.py" to train your model with multiple classifiers (it can take up to 2-3 hours). The code will pick up the best performing classifier automatically based on a model metric name which you can specify in the configuration file.

-   Add your operation scripts that handle the core scripts (e.g sending the training script to a compute target, registering a model, creating an azure ml pipeline,etc) to **operation/execution**. We provide some examples to easily setup your experiments and AML Pipeline

The project folders are structured in a way to rapidly move from a notebook experimentation to refactored code ready for deployment as following: ![design folder](https://github.com/microsoft/dstoolkit-classification-solution-accelerator/blob/main/docs/media/items.png)

General Coding Guidelines
====================================================================================================================================================

For more details on the coding guidelines and explanation on the folder structure, please go to [data/docs/how-to](https://github.com/microsoft/dstoolkit-classification-solution-accelerator/blob/main/docs/how-to/GettingStarted.md).

1.  Core scripts should receive parameters/config variables only via code arguments and must not contain any hardcoded variables in the code (like dataset names, model names, input/output path, ...). If you want to provide constant variables in those scripts, write default values in the argument parser.

2.  Variable values must be stored in ***configuration/configuration-aml.yml***. These files will be used by the execution scripts (azureml python sdk or azure-cli) to extract the variables and run the core scripts.

3.  Two distinct configuration files for environment creation:

    -   (A) for local dev/experimentation: may be stored in the project root folder (requirement.txt or environment.yml). It is required to install the project environment on a different laptop, devops agent, etc.
    -   (B) for remote compute: stored in ***configuration/environments/environment_x*** contains only the necessary Python packages and versions if speficied to be installed on remote compute targets or AKS. One subfolder is for model training and the other is for batch inference.
4.  There are only 2 core secrets to handle: the azureml workspace authentication key and a service principal. Depending on your use-case or constraints, these secrets may be required in the core scripts or execution scripts. We provide the logic to retrieve them in a ***utils.py*** file in ***src*** and a ***workspace.py*** in ***operation/execution/utils***. If you want to run the scripts locally, you need to download a config file from your AML workspace and then copy this config file into your project folder.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
