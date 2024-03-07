import numpy as np
from rdkit import Chem
import pandas as pd

'''Get normalized molecular weight of a molecule'''
def canonic_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return smiles
    return canonical_smiles

def Validity(smiles_list):
    num_valid = 0
    valid_smiles_list = np.array([])
    for smiles in smiles_list:
        if smiles=="None":
            continue
        is_valid = True
        try:
            Chem.SanitizeMol(Chem.MolFromSmiles(smiles))
        except Exception:
            is_valid = False

        if is_valid:
            num_valid += 1
            valid_smiles_list = np.append(valid_smiles_list,canonic_smiles(smiles))

    return num_valid/len(smiles_list),valid_smiles_list

def Uniquity(valid_smiles_list):
    num_smiles = valid_smiles_list.shape[0]
    if num_smiles==0:
        return 0
    sr=pd.Series(valid_smiles_list)

    return sr.unique().shape[0]/num_smiles

def Novelty(train_smiles_list,valid_smiles_list):
    num_smiles = valid_smiles_list.shape[0]
    if num_smiles==0:
        return 0
    dup=0
    for smiles in valid_smiles_list:
        dup+=train_smiles_list.apply(lambda x:1 if x==smiles else 0).sum()

    novel = valid_smiles_list.shape[0] - dup

    return novel / num_smiles

def get_fp(smiles):
    try:
        fp=Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    except:
        fp=0
    return fp

def IntDivp(valid_smiles_list,p=1):
    T=0
    fp_list=[get_fp(i) for i in valid_smiles_list]
    cnt=0
    for i in fp_list:
        for j in fp_list:
            if i==0 or j==0:
                continue
            else:
                T+=np.power(Chem.DataStructs.FingerprintSimilarity(i,j),p)
                cnt+=1

    return 1-np.power(T/cnt,1/p)
