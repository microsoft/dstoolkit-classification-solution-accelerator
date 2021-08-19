# Set up an input dataset

To setup your own input dataset for your classification project in you azure subscription, follow these steps:

1. Get your data file ready.  Or download "two_class.csv" file from the home folder of this repo if you don't have your own data.  Your input file can be locally or in Azure Blob storage (you can set up a Datastore for it).
2. Launch the Azure Machine Learning studio which you had just deployed using "PIPELINE-0-setup.yml".
3. Inside your AML studio, create a new tabular dataset from local files using your CSV file or the downloaded CSV file.
4. Add the new registered dataset name and other variables (model name, etc) to **_configuration/configuration-aml.variables.yml_**

