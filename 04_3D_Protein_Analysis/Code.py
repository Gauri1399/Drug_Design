#Run only once
#pip install biopython scikit-learn pandas numpy py3Dmol
import requests

# Set the URL
url = "https://files.rcsb.org/download/1BRS.pdb"

# Get response from URL
response = requests.get(url)

# Check if the response is successful
if response.status_code == 200:
    # Open file with write permission and save the content
    with open("1BRS.pdb", "w") as file:
        file.write(response.text)
    print("File downloaded and saved as 1BRS.pdb")
else:
    # Print Download failed with reason
    print(f"Failed to download file. Status code: {response.status_code}")

from Bio.PDB import PDBParser

pdb_file = "1BRS.pdb"
parser = PDBParser(QUIET=True)
structure = parser.get_structure("1BRS", pdb_file)

model = next(structure.get_models(), None)
if model is None:
    raise ValueError("No models found in structure.")

# Collect all chains in the first model
chains = {ch.id: ch for ch in model.get_chains()}
print("Found chains:", list(chains.keys()))

# Basic per-chain summary
for cid, ch in chains.items():
    residues = [res for res in ch]
    print(f"Chain {cid}: {len(residues)} residues")

from Bio.PDB import NeighborSearch

def get_interface_residues(chain_query, chain_target, distance_threshold=4.0):
    """Return a list of residues in chain_query that are within distance_threshold Å
    of any atom in chain_target."""
    target_atoms = list(chain_target.get_atoms())
    ns = NeighborSearch(target_atoms)
    interface_residues = set()
    for residue in chain_query:
        for atom in residue.get_atoms():
            # Biopython stores coords as numpy arrays; both .coord and .get_coord() are common
            coord = getattr(atom, "coord", None)
            if coord is None:
                coord = atom.get_coord()
            if ns.search(coord, distance_threshold):
                interface_residues.add(residue)
                break
    return list(interface_residues)

# Compute interface residues for all ordered chain pairs (i, j), i != j
all_interfaces = {}
for cq in chains:
    for ct in chains:
        if cq == ct:
            continue
        res_list = get_interface_residues(chains[cq], chains[ct], distance_threshold=4.0)
        all_interfaces[(cq, ct)] = res_list

# Summary table of counts
import pandas as pd
interface_counts = []
for (cq, ct), residues in all_interfaces.items():
    interface_counts.append({
        "Chain_Query": cq,
        "Chain_Target": ct,
        "NumInterfaceResidues": len(residues)
    })
df_interface_counts = pd.DataFrame(interface_counts).sort_values(
    ["Chain_Query", "Chain_Target"]
).reset_index(drop=True)
df_interface_counts

import numpy as np

# Kyte–Doolittle hydrophobicity (representative dictionary)
hydrophobicity = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

def compute_features_from_residues(residues):
    hydros = []
    net_charge = 0.0
    for res in residues:
        name = res.get_resname().strip()
        if name in hydrophobicity:
            hydros.append(hydrophobicity[name])
        # Charge heuristic
        if name in ["ARG", "LYS"]:
            net_charge += 1
        elif name == "HIS":
            net_charge += 0.5
        elif name in ["ASP", "GLU"]:
            net_charge -= 1
    return {
        "NumResidues": len(residues),
        "AvgHydrophobicity": float(np.mean(hydros)) if hydros else 0.0,
        "NetCharge": net_charge
    }

# Build a features table for every (query, target) interface
rows = []
for (cq, ct), residues in all_interfaces.items():
    feats = compute_features_from_residues(residues)
    rows.append({
        "Chain_Query": cq,
        "Chain_Target": ct,
        "NumInterfaceResidues": feats["NumResidues"],
        "AvgHydrophobicityInterface": feats["AvgHydrophobicity"],
        "InterfaceNetCharge": feats["NetCharge"]
    })
df_interface_features = pd.DataFrame(rows).sort_values(
    ["Chain_Query", "Chain_Target"]
).reset_index(drop=True)
df_interface_features

def select_central_pocket(chain, pocket_size=11):
    """Select a central window of residues by residue number as a simple pocket proxy."""
    resnums = [res.id[1] for res in chain]
    if not resnums:
        return []
    resnums_sorted = sorted(resnums)
    mid = len(resnums_sorted) // 2
    half = pocket_size // 2
    window = set(resnums_sorted[max(0, mid - half): min(len(resnums_sorted), mid + half + 1)])
    pocket = [res for res in chain if res.id[1] in window]
    return pocket

# Compute pocket features for each chain
pocket_rows = []
for cid, ch in chains.items():
    pocket = select_central_pocket(ch, pocket_size=11)  # editable
    feats = compute_features_from_residues(pocket)
    pocket_rows.append({
        "Chain": cid,
        "NumPocketResidues": feats["NumResidues"],
        "AvgHydrophobicityPocket": feats["AvgHydrophobicity"],
        "PocketNetCharge": feats["NetCharge"]
    })
df_pocket_features = pd.DataFrame(pocket_rows).sort_values("Chain").reset_index(drop=True)
df_pocket_features
# Merge interface & pocket features by choosing a particular (query, target) pair per row.
# Here we will create a few demo rows.
import random

def random_row():
    # pick a random interface pair to combine with a random pocket chain
    if not len(df_interface_features):
        raise ValueError("No interface features computed.")
    if not len(df_pocket_features):
        raise ValueError("No pocket features computed.")
    iface = df_interface_features.sample(1, random_state=random.randint(0, 9999)).iloc[0].to_dict()
    pocket = df_pocket_features.sample(1, random_state=random.randint(0, 9999)).iloc[0].to_dict()
    row = {
        "Chain_Query": iface["Chain_Query"],
        "Chain_Target": iface["Chain_Target"],
        "NumInterfaceResidues": iface["NumInterfaceResidues"],
        "AvgHydrophobicityInterface": iface["AvgHydrophobicityInterface"],
        "InterfaceNetCharge": iface["InterfaceNetCharge"],
        "PocketChain": pocket["Chain"],
        "NumPocketResidues": pocket["NumPocketResidues"],
        "AvgHydrophobicityPocket": pocket["AvgHydrophobicityPocket"],
        "PocketNetCharge": pocket["PocketNetCharge"],
        # Arbitrary label for demo; replace with real labels in practice.
        "Activity": random.choice([0, 1])
    }
    return row

demo_rows = [random_row() for _ in range(12)]
df = pd.DataFrame(demo_rows)
df
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Encode chain IDs as categorical for ML (simple one-hot)
X = pd.get_dummies(df.drop("Activity", axis=1), columns=["Chain_Query", "Chain_Target", "PocketChain"])
y = df["Activity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred, zero_division = 0))
import py3Dmol
from collections import defaultdict

requested_pairs = [
    ("A","D"), ("A","E"), ("A","F"),
    ("B","D"), ("B","E"), ("B","F"),
    ("C","D"), ("C","E"), ("C","F"),
]

pair_views_info = []
for cq, ct in requested_pairs:
    if cq not in chains or ct not in chains:
        print(f"Skipping {cq}-{ct}: missing chain(s)")
        continue

    interface = get_interface_residues(chains[cq], chains[ct], distance_threshold=5.0)
    interface_resnums = sorted({res.id[1] for res in interface})
    pocket = select_central_pocket(chains[cq], pocket_size=11)
    pocket_resnums = sorted({res.id[1] for res in pocket})

    with open("1BRS.pdb", "r") as fh:
        pdb_text = fh.read()

    view = py3Dmol.view(width=800, height=520)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "lightgrey"}})

    # Interface (red) — will show only if there are contacting residues
    for r in interface_resnums:
        view.addStyle({"chain": cq, "resi": str(r)}, {"stick": {"color": "red"}})

    # Pocket on query chain (blue) — always shown
    for r in pocket_resnums:
        view.addStyle({"chain": cq, "resi": str(r)}, {"stick": {"color": "blue"}})

    view.zoomTo()
    print(f"Viewer (Query={cq}, Target={ct}) – interface_residues={len(interface_resnums)}")
    view.show()
    pair_views_info.append((cq, ct, len(interface_resnums)))
