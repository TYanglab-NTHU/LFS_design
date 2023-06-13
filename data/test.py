from rdkit import Chem
import csv
transsmis = set()
with open('/home/scorej41075/Workshop/data/trainsmis-kekulized.smi','r') as f1:
    smiles = f1.read().splitlines()

for smi in smiles:
    try:
        mol = Chem.MolFromSmiles(smi)
        Chem.Kekulize(mol)
        smile = Chem.MolToSmiles(mol,isomericSmiles=True)
        transsmis.add(smile)
    except:
        pass
    

foutsmi = [[line] for line in transsmis]
with open('/home/scorej41075/Workshop/data/trainsmis.smi','w') as f1:
    writer = csv.writer(f1)
    writer.writerows(foutsmi)