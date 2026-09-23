# AI-Driven Drug Failure Analysis & Structural Intelligence Pipeline

## Overview

This project is a Python-based bioinformatics pipeline that combines biomedical database mining, natural language processing (NLP), and structural bioinformatics to investigate drug candidates associated with clinical trial failure and examine their molecular targets.

The pipeline integrates:

* **PubChem** for compound identification and metadata retrieval
* **PubMed/NCBI Entrez** for literature retrieval related to clinical trial outcomes
* **spaCy** for NLP-based extraction of failure-related information from abstracts
* **Keyword analysis and visualization** to identify recurring drug-failure signals
* **UniProt** for target protein identification
* **AlphaFold Protein Structure Database** for predicted protein structures
* **py3Dmol** for interactive visualization of protein structures

The project demonstrates an end-to-end approach for combining literature mining, biomedical data retrieval, NLP, and structural analysis to investigate patterns associated with drug development failures.

## Features

* Programmatic retrieval of drug metadata from PubChem
* Automated PubMed literature searches using NCBI Entrez
* NLP-based analysis of clinical trial abstracts
* Extraction and visualization of frequently occurring failure-related terms
* Retrieval of protein structures using UniProt identifiers
* Integration with AlphaFold structural predictions
* Interactive 3D visualization of protein structures using py3Dmol

## Workflow

```text
Drug Name
    ↓
PubChem
    ↓
Compound Metadata
    ↓
PubMed / NCBI Entrez
    ↓
Clinical Trial Literature
    ↓
spaCy NLP Analysis
    ↓
Failure-Related Terms & Signals
    ↓
Target Protein UniProt ID
    ↓
AlphaFold Structure
    ↓
Interactive 3D Visualization
```

## Installation

Make sure Python 3 is installed, then install the required dependencies:

```bash
pip install biopython py3Dmol spacy matplotlib requests
python -m spacy download en_core_web_sm
```

## Usage

1. Update the `main()` function in `Code.py` with:

   * **Drug name** — the compound you want to investigate
   * **Target protein UniProt ID** — the UniProt identifier for the target protein

2. Run the pipeline:

```bash
python Code.py
```

3. The pipeline will:

   * Search PubChem for compound metadata
   * Query PubMed for literature related to clinical trial failure
   * Process retrieved abstracts using spaCy
   * Identify and visualize frequently occurring failure-related terms
   * Retrieve the target protein structure from the AlphaFold Protein Structure Database
   * Generate an interactive 3D visualization of the protein structure

## Technologies

* Python
* Biopython
* NCBI Entrez / PubMed
* PubChem
* spaCy
* UniProt
* AlphaFold Protein Structure Database
* py3Dmol
* Matplotlib
* Requests

## References

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature, 596*(7873), 583–589. https://doi.org/10.1038/s41586-021-03819-2

Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength natural language processing in Python. Explosion AI. https://spacy.io/

National Center for Biotechnology Information (NCBI). Entrez Programming Utilities. National Library of Medicine. https://www.ncbi.nlm.nih.gov/books/NBK25500/

National Center for Biotechnology Information (NCBI). PubChem. National Library of Medicine. https://pubchem.ncbi.nlm.nih.gov/

Varadi, M., Anyango, S., Deshpande, M., Nair, S., Natassia, C., Yordanova, G., et al. (2022). AlphaFold Protein Structure Database: Massively expanding the structural coverage of protein-sequence space. *Nucleic Acids Research, 50*(D1), D439–D444. https://doi.org/10.1093/nar/gkab1061
