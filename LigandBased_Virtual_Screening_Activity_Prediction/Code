!pip install rdkit biopandas numpy pandas matplotlib scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from rdkit import Chem
from rdkit.Chem import Descriptors

smiles_list = input("Enter a list of SMILES strings separated by commas: ").split(',')
smiles_list = [s.strip() for s in smiles_list]

molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def compute_descriptors(mol):
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol)
        ]
    else:
        return [None, None, None]

descriptor_data = [compute_descriptors(mol) for mol in molecules]
df = pd.DataFrame(descriptor_data, columns=['MolWt', 'NumHDonors', 'NumHAcceptors'])
df['SMILES'] = smiles_list

df.head()
df['Docking_Score'] = np.random.uniform(-9, -4, len(df))
df.sort_values(by='Docking_Score', ascending=True).head()

df['Active'] = df['Docking_Score'] < -6.5

X = df[['MolWt', 'NumHDonors', 'NumHAcceptors']]
y = df['Active']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
