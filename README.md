
# Crop Recommendation System (oct30_crop_ds.ipynb)

## Overview

This Jupyter notebook implements a machine learning-based crop recommendation system using a dataset containing features related to soil and weather conditions. The notebook covers the full data science workflow, including data loading, preprocessing, model training, evaluation, and saving the best-performing model for future use.[1]

## Dataset

- The dataset, `Crop_recommendation.csv`, contains 2,200 samples and 8 columns:
  - **N, P, K**: Nitrogen, Phosphorus, Potassium content in soil
  - **temperature**: Average temperature (°C)
  - **humidity**: Relative humidity (%)
  - **ph**: Soil pH value
  - **rainfall**: Rainfall (mm)
  - **label**: Recommended crop[1]

## Main Objective

The main goal is to predict the most suitable crop for given environmental conditions and soil nutrients using different classification algorithms.[1]

## Workflow

- **Import Libraries**: Utilizes pandas, numpy, matplotlib, seaborn, and scikit-learn.[1]
- **Data Loading and Exploration**: Reads and explores the dataset, inspecting data types and summary statistics.[1]
- **Preprocessing**: Splits the data into input features (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`) and target (`label`). Uses an 80-20 train-test split.[1]
- **Model Training**: Trains four classification models:
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)[1]
- **Evaluation**: Evaluates models using accuracy and classification reports. Displays detailed performance metrics for each class and model, achieving high accuracy (up to 99%).[1]
- **Model Saving**: Saves the best-performing model (Naive Bayes) as `bestlomodel.pkl`.[1]

## Requirements

Install the following Python packages before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Place the dataset CSV file in the same directory.
2. Open and run the notebook cell by cell.
3. The notebook will show which crop is best for any given sample of provided conditions.[1]

## Output

- Shows classification metrics (precision, recall, F1-score, accuracy) for all four models.
- The best model can be used for later predictions by loading `bestlomodel.pkl`.[1]

***

This README provides a clear overview, setup instructions, and an explanation of the workflow for new users or collaborators working with your notebook.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/97486039/0e8a97fa-b25d-4c5b-98a5-88cf64caeda9/oct30_crop_ds.ipynb)
