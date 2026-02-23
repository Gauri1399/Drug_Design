
from rdkit import Chem
from rdkit.Chem import Descriptors

smiles_list = ["CCO", "CCCC", "CCN(CC)CC", "C1=CC=CC=C1O"]  # ethanol, butane, triethylamine, phenol
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    print(smi, {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    })

from rdkit import DataStructs
from rdkit.Chem import AllChem

smiles = ["CCO", "CCCO", "CCCCO", "CCCN"]  # homologous alcohols and an amine
mols = [Chem.MolFromSmiles(s) for s in smiles]
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]

print("Tanimoto similarity matrix:")
for i in range(len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    print(smiles[i], [round(s, 2) for s in sims])

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

y_true = np.array([5.1, 4.8, 4.3, 3.9, 3.5])
y_pred1 = np.array([5.0, 4.9, 4.4, 4.0, 3.4])  # Model A
y_pred2 = np.array([5.2, 5.0, 4.0, 3.7, 3.0])  # Model B

for name, pred in {"Model A": y_pred1, "Model B": y_pred2}.items():
    r2 = r2_score(y_true, pred)
    rmse = mean_squared_error(y_true, pred, squared=False)
    mae = mean_absolute_error(y_true, pred)
    print(f"{name}: R²={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.seed(42)
X = np.random.rand(30, 5)
y_true = 3*X[:,0] - 2*X[:,1] + np.random.normal(0, 0.1, 30)

model = LinearRegression()
true_score = cross_val_score(model, X, y_true, cv=5, scoring='r2').mean()

y_rand = np.random.permutation(y_true)
rand_score = cross_val_score(model, X, y_rand, cv=5, scoring='r2').mean()

print(f"True Q²: {true_score:.3f}, Y-Randomized Q²: {rand_score:.3f}")
