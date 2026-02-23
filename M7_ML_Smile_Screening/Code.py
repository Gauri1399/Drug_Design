# ----------------------------
# SmileML: SMILES-Based Molecule Activity Prediction
# ----------------------------

# Install dependencies (only needed if not installed)
# !pip install rdkit biopandas numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error

# ----------------------------
# Step 1: SMILES input and descriptor calculation
# ----------------------------
smiles_list = input("Enter SMILES strings separated by commas: ").split(',')
smiles_list = [s.strip() for s in smiles_list]

molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]

def compute_descriptors(mol):
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol)
        ]
    else:
        return [None]*5

descriptor_data = [compute_descriptors(mol) for mol in molecules]
df = pd.DataFrame(descriptor_data, columns=['MolWt', 'NumHDonors', 'NumHAcceptors', 'LogP', 'TPSA'])
df['SMILES'] = smiles_list

print("\nDescriptor Table:")
print(df)

# ----------------------------
# Step 2: Fingerprint similarity
# ----------------------------
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]

print("\nTanimoto similarity matrix:")
for i in range(len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    print(smiles_list[i], [round(s, 2) for s in sims])

# ----------------------------
# Step 3: Simulate docking scores & define activity
# ----------------------------
np.random.seed(42)
df['Docking_Score'] = np.random.uniform(-9, -4, len(df))
df['Active'] = df['Docking_Score'] < -6.5

print("\nDocking Scores & Activity:")
print(df[['SMILES', 'Docking_Score', 'Active']])

# ----------------------------
# Step 4: Random Forest classification
# ----------------------------
X = df[['MolWt', 'NumHDonors', 'NumHAcceptors', 'LogP', 'TPSA']]
y = df['Active']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ----------------------------
# Step 5: Example regression evaluation 
# ----------------------------
y_true = np.array([5.1, 4.8, 4.3, 3.9, 3.5])
y_pred1 = np.array([5.0, 4.9, 4.4, 4.0, 3.4])
y_pred2 = np.array([5.2, 5.0, 4.0, 3.7, 3.0])

print("\nRegression Metrics for Example Models:")
for name, pred in {"Model A": y_pred1, "Model B": y_pred2}.items():
    r2 = r2_score(y_true, pred)
    rmse = mean_squared_error(y_true, pred, squared=False)
    mae = mean_absolute_error(y_true, pred)
    print(f"{name}: R²={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

# ----------------------------
# Step 6: Y-randomization / Q² validation 
# ----------------------------
np.random.seed(42)
X_reg = np.random.rand(30, 5)
y_reg = 3*X_reg[:,0] - 2*X_reg[:,1] + np.random.normal(0, 0.1, 30)

lin_model = LinearRegression()
true_score = cross_val_score(lin_model, X_reg, y_reg, cv=5, scoring='r2').mean()
y_rand = np.random.permutation(y_reg)
rand_score = cross_val_score(lin_model, X_reg, y_rand, cv=5, scoring='r2').mean()

print(f"\nTrue Q²: {true_score:.3f}, Y-Randomized Q²: {rand_score:.3f}")
