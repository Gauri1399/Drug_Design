# SmileML – SMILES-Based Molecule Activity Prediction with ML

## Project Overview

**SmileML** is a Python-based cheminformatics and machine learning workflow for analyzing small molecules from **SMILES representations**. The workflow calculates molecular descriptors, generates molecular fingerprints, evaluates structural similarity, and applies machine learning to classify compounds based on activity labels.

The project demonstrates an end-to-end workflow for **molecular feature engineering, similarity analysis, activity classification, and model validation**.

## Features

* Accepts molecules as **SMILES strings**
* Calculates molecular descriptors using RDKit:

  * Molecular Weight (MolWt)
  * LogP
  * Topological Polar Surface Area (TPSA)
  * Hydrogen-bond donors
  * Hydrogen-bond acceptors
* Generates **Morgan fingerprints**
* Calculates pairwise **Tanimoto similarity**
* Supports activity labels derived from simulated docking scores or experimental activity data
* Trains a **Random Forest classifier** for activity prediction
* Generates classification metrics
* Supports regression evaluation using R², RMSE, and MAE when continuous activity values are available
* Supports **Y-randomization** to assess whether model performance may arise from chance
* Can be extended with additional molecular descriptors, fingerprints, and machine learning models

## How It Works

### 1. SMILES Input

Provide a list of chemical compounds represented as SMILES strings.

### 2. Molecular Descriptor Calculation

RDKit is used to calculate physicochemical descriptors for each compound, including molecular weight, LogP, TPSA, and hydrogen-bonding properties.

### 3. Molecular Fingerprinting

Morgan fingerprints are generated to represent molecular structures and calculate pairwise **Tanimoto similarity** between compounds.

### 4. Activity Label Generation

Compounds can be assigned active/inactive labels using simulated docking scores for demonstration purposes or using experimentally measured activity data when available.

### 5. Machine Learning

A Random Forest classifier is trained using molecular descriptors to predict compound activity.

### 6. Model Validation

Model performance can be evaluated using classification metrics. For continuous activity prediction, regression metrics such as **R², RMSE, and MAE** can also be calculated.

Y-randomization can be performed by randomly shuffling activity labels and retraining the model to assess whether predictive performance is substantially different from randomized labels.

## Outputs

The workflow produces:

* Molecular descriptors for each input SMILES
* Morgan fingerprints
* Tanimoto similarity matrix
* Activity labels
* Random Forest classification results
* Model performance metrics
* Optional regression metrics
* Optional Y-randomization results

## Installation

Make sure Python 3.x is installed.

Clone the repository:

```bash
git clone https://github.com/Gauri1399/05_ML_Smile_Screening.git
cd 05_ML_Smile_Screening
```

Install the required packages:

```bash
pip install rdkit biopandas numpy pandas matplotlib scikit-learn
```

## Technologies

* Python
* RDKit
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Morgan fingerprints
* Tanimoto similarity
* Random Forest

## Repository

[View the source code on GitHub](https://github.com/Gauri1399/05_ML_Smile_Screening)
