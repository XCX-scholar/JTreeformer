import numpy as np
import pandas as pd
import rdkit.Chem as Chem

folder="path/to/dataset"

def gen_mrnn_data(num_valid,source_path="ZINC2.csv",train_path="zinc_test.smi",valid_path="zinc_train.smi"):
    smiles=pd.read_csv(folder+source_path)["smiles"]
    size=smiles.shape[0]
    idx=np.random.choice(range(size),num_valid,replace=False)
    valid=smiles[idx].values
    tgt_file=open(folder+train_path, 'a')
    for i in valid:
        tgt_file.write(i+ '\n')
    tgt_file.close()
    tgt_file=open(folder+valid_path, 'a')
    for i in range(size):
        if i not in idx:
            tgt_file.write(smiles[i] + '\n')
    tgt_file.close()