# Protein Interface & Pocket Analysis with ML and 3D Visualization

## Project Overview

This Python-based bioinformatics workflow analyzes protein structures from the **Protein Data Bank (PDB)** to identify structural regions associated with protein function. The pipeline combines **protein structure analysis, protein-protein interface detection, pocket feature extraction, machine learning, and interactive 3D visualization**.

The workflow can be used to characterize potential functional regions and prioritize interfaces or pockets for further structural analysis.

## Key Features

* **PDB Structure Retrieval:** Download and parse protein structures directly from the PDB.
* **Interface Residue Detection:** Identify residues participating in protein-protein interfaces using a configurable distance threshold.
* **Pocket Feature Extraction:** Calculate structural and physicochemical features, including hydrophobicity, net charge, and residue composition.
* **Machine Learning Classification:** Train a Random Forest classifier using extracted structural features to classify interfaces or pockets based on labeled data.
* **3D Structural Visualization:** Generate interactive protein structure visualizations with `py3Dmol`, highlighting interface and pocket residues.

## Workflow

```text
PDB Structure
     ↓
Structure Parsing
     ↓
Interface Residue Detection
     ↓
Pocket Identification
     ↓
Feature Extraction
     ↓
Random Forest Classification
     ↓
3D Visualization
```

## Structural Analysis

The workflow extracts information from protein structures including:

* Protein chains and residues
* Protein-protein interface residues
* Pocket residue composition
* Hydrophobicity
* Net charge
* Residue counts
* Spatial relationships between structural regions

Interface residues are identified using a configurable distance threshold between atoms or residues from interacting chains.

## Machine Learning

A **Random Forest classifier** is used to analyze extracted structural features and classify protein interfaces or pockets according to their functional labels.

The feature set can include:

* Hydrophobic residue content
* Charged residue content
* Residue composition
* Pocket size
* Interface characteristics

The workflow can be extended with additional structural descriptors and experimentally annotated datasets.

## 3D Visualization

Interactive protein structures are generated using **py3Dmol**.

The visualization highlights:

* **Interface residues** in red
* **Pocket residues** in blue
* Protein structure and chain organization for structural context

This allows structural features identified computationally to be examined directly within the three-dimensional protein structure.

## Installation

Clone the repository:

```bash
git clone https://github.com/Gauri1399/04_3D_Protein_Analysis.git
cd 04_3D_Protein_Analysis
```

Install the required Python packages:

```bash
pip install biopython numpy pandas scikit-learn py3Dmol requests
```

## Usage

1. Provide a valid PDB identifier or protein structure.
2. Run the structural analysis workflow.
3. Detect protein-protein interface residues and potential pockets.
4. Extract structural and physicochemical features.
5. Train or apply the Random Forest classifier using labeled data.
6. Generate an interactive 3D visualization of the analyzed structure.

## Technologies

* Python
* Biopython
* PDB
* NumPy
* Pandas
* Scikit-learn
* Random Forest
* py3Dmol

## Repository

[View the source code on GitHub](https://github.com/Gauri1399/04_3D_Protein_Analysis)
