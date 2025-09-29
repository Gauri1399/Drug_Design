# Import required libraries
from Bio import Entrez  # To interact with PubMed and PubChem
import py3Dmol           # For 3D molecular visualization
import requests          # For fetching AlphaFold structures
import spacy             # Natural Language Processing
from collections import Counter  # For word frequency analysis
import matplotlib.pyplot as plt  # For plotting

# Function to search for a compound in PubChem using drug name
def search_pubchem(drug_name):
    handle = Entrez.esearch(db="pccompound", term=drug_name, retmax=1)
    record = Entrez.read(handle)
    handle.close()

    if record["IdList"]:
        compound_id = record["IdList"][0]
        return compound_id
    else:
        return None

# Function to search PubMed for clinical trials related to drug failures
def search_pubmed_for_failed_trials(drug_name):
    # Search term includes trial failure terms
    search_term = drug_name + " AND clinical trial[tiab] AND (failure OR withdrawn OR terminated)"
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=5)
    record = Entrez.read(handle)
    handle.close()

    if record["IdList"]:
        print("PubMed Articles on Failed Clinical Trials for", drug_name, ":", record['IdList'])
        return record["IdList"]
    else:
        return []

# Function to fetch abstracts from PubMed given article IDs
def fetch_pubmed_abstracts(article_ids):
    if not article_ids:
        print("No abstracts found.")
        return []

    handle = Entrez.efetch(db="pubmed", id=",".join(article_ids), rettype="abstract", retmode="text")
    abstracts = handle.read()
    handle.close()

    print("\n Full Abstracts from PubMed:")
    print(abstracts)
    return abstracts

# Function to extract failure-related terms using spaCy
def analyze_failure_reasons(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Keywords commonly associated with clinical trial failures
    failure_keywords = ["toxicity", "adverse", "mortality", "side effects", "withdrawn", "terminated", "FDA rejection"]

    # Count the occurrence of keywords
    word_freq = Counter(token.text.lower() for token in doc if token.text.lower() in failure_keywords)

    print("\n AI-Extracted Failure Reasons:")
    for word, freq in word_freq.items():
        print(word, ":", freq, "occurrences")

    # Plot the frequency of failure reasons
    plt.figure(figsize=(8, 5))
    plt.bar(word_freq.keys(), word_freq.values(), color="blue")
    plt.title("Failure Reason Frequency in PubMed Abstracts")
    plt.xlabel("Failure Keywords")
    plt.ylabel("Frequency")
    plt.show()

# Function to fetch AlphaFold protein structure using UniProt ID
def fetch_alphafold_structure(uniprot_id):
    alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response = requests.get(alphafold_url)

    if response.status_code == 200:
        print("AlphaFold PDB file found for UniProt ID:", uniprot_id)
        return response.text  # Return PDB structure as text
    else:
        print("Structure not found for UniProt ID:", uniprot_id)
        return None

# Function to visualize the 3D protein structure using py3Dmol
def visualize_alphafold_structure(pdb_content):
    if not pdb_content:
        print("No structure available for visualization.")
        return

    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(pdb_content, "pdb")
    viewer.setStyle({"cartoon": {"color": "spectrum"}})
    viewer.zoomTo()
    return viewer.show()

# Main driver function
def main():
    # REQUIRED: Set your email for Entrez API access
    Entrez.email = "your_email@example.com"

    # -------------------- USER INPUTS --------------------
    drug_name = "torcetrapib"  # Example: A failed cholesterol drug
    target_uniprot_id = "P04150"  # Example: UniProt ID of target protein (e.g., CETP)
    # -----------------------------------------------------

    # Step 1: Search for the compound in PubChem
    compound_id = search_pubchem(drug_name)
    if compound_id:
        print(f"PubChem Compound ID for {drug_name}: {compound_id}")
    else:
        print("Compound not found in PubChem.")

    # Step 2: Search PubMed for failed clinical trials
    failed_trial_articles = search_pubmed_for_failed_trials(drug_name)

    # Step 3: Fetch abstracts and analyze reasons for failure
    if failed_trial_articles:
        abstracts = fetch_pubmed_abstracts(failed_trial_articles)
        analyze_failure_reasons(abstracts)

    # Step 4: Fetch and visualize the target protein structure
    pdb_content = fetch_alphafold_structure(target_uniprot_id)
    if pdb_content:
        visualize_alphafold_structure(pdb_content)

# Execute script
if __name__ == "__main__":
    main()
