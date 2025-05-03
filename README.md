You are are reading the README file which has 3 parts :
1. Summarizes all the steps to follow to replicate and use this project.
2. Highlight the .pynb notebook that has all the steps involved in this project
3. Some highlight of insights and reasoning behind this approach of the project

------------------### Part 1 ###----------------------

The user can refer to the directory named 'eda visualizations + report' to understand the data through Exploratory data analysis. In same directory there is a report based on the EDA.

Before begining any of the scripting process, the user has to first to install the dependencies: pip install -r requirements.txt

1. Load the dataset

The CIC-IDS dataset is very big (few hundreds of GB). We have extracted the relevant information from it. We have 20 pqarquet files that have been preprocessed already for this project.
The user needs to run the file 'dataload.py' in the data folder which will export a csv file that could be used for Machine learning processes.
^^ Please note that in the output data folder we already have the .zip version f the file (the file was zipped due to Git Storage constrains'

2. Pre-processing

To pre-process and clean the file from outliers (too long payload information), please use the following script:
python -m preprocessing_for_models.preprocessing_model1

3. Tokenizing of the payload text for NLP process

To tokenize the payload text, please use the following script:
python -m preprocessing_for_models.tokenizer_model1

4. to build and train a model

To train a model, use the following script:
python -m models.final_multi_input

#If not the user can user pre-trained model that has been saved as 'final_multi_input_model.zip'. After unzip, the user can directly load the model and deploy.

5. Evaluate a model

To evaluate a model after training or loading, use the following script:
python -m models.model_evaluation

You can add your own arguments to chose the model, the batch size, and the epoch for example

------------------### Part 2 ###----------------------

Python Notebook that can opened in environments such as Anaconda Jupyter notebook oe Google Collab
> Google collab will have all of dependecies pre installed
> If the user wants to run the notebook in local Jupyter notebook environments, the following packages are required:
-- Pandas
-- Scikit-learn
-- keras

------------------### Part 3 ###----------------------

Exploratory data analysis revealed key distinctions between benign and malicious payloads. The imbalance in class distribution, strong entropy correlations, distinct header bit patterns, and divergent textual content provide valuable signals for designing robust classification models. Addressing class imbalance and emphasizing high-variance features will be crucial in the next stages of modeling and feature engineering.

Why multi-input neural network model ? 
> We are working with 3 heterogeneous input types: structured, numerical, and text. A single model can’t handle all 3 well unless you separate inputs.
> A simple model (like logistic regression or one big Dense net) might overfit fast or miss nuances across modalities.
> Multi-branch architectures are common in cybersecurity ML; e.g., detecting phishing, botnet traffic, malware signatures where multiple signals are fused.
