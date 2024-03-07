import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from jtnn_utils.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from jtnn_utils.vocab import *
import sys
import pickle
import argparse


class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)
        self.hs=sum([a.GetTotalNumHs() for a in self.mol.GetAtoms()])

        self.clique = [x for x in clique]  # copy
        self.neighbors = []
        self.nid=-1
        self.is_leaf=True

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:  # Leaf node, no need to mark
                continue
            for cidx in nei_node.clique:
                # allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands, aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i, cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0: cands = new_cands

        if len(cands) > 0:
            self.cands, _ = list(zip(*cands))
            self.cands = list(self.cands)
        else:
            self.cands = []


class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        # Stereo Generation (currently disabled)
        # mol = Chem.MolFromSmiles(smiles)
        # self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        # self.smiles2D = Chem.MolToSmiles(mol)
        # self.stereo_cands = decode_stereo(self.smiles2D)

        cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        self.w = Descriptors.MolWt(self.mol)
        self.logp = Descriptors.MolLogP(self.mol)
        self.tpsa = Descriptors.TPSA(self.mol)
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0: root = i

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1:  # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()


def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx:
            continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


# def main_mol_tree(oinput, ovocab, MAX_TREE_WIDTH=50):
#     cset = set()
#     with open(oinput, 'r') as input_file:
#         for i, line in enumerate(input_file.readlines()):
#             smiles = line.strip().split()[0]
#             alert = False
#             mol = MolTree(smiles)
#             for c in mol.nodes:
#                 if c.mol.GetNumAtoms() > MAX_TREE_WIDTH:
#                     alert = True
#                 cset.add(c.smiles)
#             if len(mol.nodes) > 1 and alert:
#                 sys.stderr.write('[WARNING]: %d-th molecule %s has a high tree-width.\n' % (i + 1, smiles))
#
#     with open(ovocab, 'w') as vocab_file:
#         for x in cset:
#             vocab_file.write(x + '\n')
#
#
# if __name__ == "__main__":
#     lg = rdkit.RDLogger.logger()
#     lg.setLevel(rdkit.RDLogger.CRITICAL)
#     sys.stderr.write('Running tree decomposition on the dataset')
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", dest="input")
#     parser.add_argument("-v", "--vocab", dest="vocab")
#     opts = parser.parse_args()
#
#     main_mol_tree(opts.input, opts.vocab)

def convert(ismiles, MAX_TREE_WIDTH=50):
    cset = set()
    i,smiles=ismiles
    alert = False
    try:
        moltree = MolTree(smiles)
        moltree.recover()
        for c in moltree.nodes:
            if c.mol.GetNumAtoms() > MAX_TREE_WIDTH:
                alert = True
            cset.add(c.smiles)
        del moltree.mol
        for node in moltree.nodes:
            del node.mol

        if len(moltree.nodes) > 1 and alert:
            sys.stderr.write('[WARNING]: %d-th molecule %s has a high tree-width.\n' % (i + 1, smiles))
        return (moltree,cset)
    except:
        print(i,smiles)
        return (None,None)
    if i % 1000==0:
        print(i)

def main_mol_tree(pool,oinput, ovocab):
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"MolDataSet"))
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)
    all_data = pool.map(convert,enumerate(oinput))
    Trees,csets=zip(*all_data)
    # print(type(Tree),type(csets))
    # print(len(Tree),len(csets))
    # print(Tree[0].smiles)
# guacamol_v1_train.smiles

    Trees=[t for t in Trees if t is not None]
    csets= [c for c in csets if c is not None]
    print(len(Trees))
    with open(os.path.join(save_path,"valid_guaca_mol_tree.pkl"), 'wb') as file:
        pickle.dump(Trees,file)
    vocab=set()
    for cset in csets:
        vocab=vocab|cset
    with open(ovocab,"a") as file:
        for v in vocab:
            file.write(v+'\n')

# 1497218
# if __name__ == "__main__":
    # import pandas as pd
    # df=pd.read_csv(r"D:\MolDataSet\test.txt")['SMILES']
    # file_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..',"guaca_data","guacamol_v1_valid.smiles"))
    # with open(file_path, 'r') as file:
    #     smiles=[x.strip() for x in file.readlines()]
    # lg = rdkit.RDLogger.logger()
    # lg.setLevel(rdkit.RDLogger.CRITICAL)
    # sys.stderr.write('Running tree decomposition on the dataset')
    # from multiprocessing import Pool
    # import os
    # pool = Pool(os.cpu_count())
    # voc_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"vocab"))
    # if os.path.isdir(voc_path) is False:
    #     os.makedirs(voc_path)
    # main_mol_tree(pool,smiles,os.path.join(voc_path,'guaca_vocab.txt'))
    # save_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"MolDataSet","valid_guaca_mol_tree.pkl"))
    # with open(save_path, 'rb') as file:
    #     data=pickle.load(file)
    # n=len(data)
    # num_nodes=0
    # max=0
    # min=0
    # for tree in data:
    #     num_node=len(tree.nodes)
    #     num_nodes+=num_node
    #     if num_node>max:
    #         max=num_node
    #     if num_node<min:
    #         min=num_node
    # print(min,max,n,num_nodes/n)
# 0 28 1497218 13.474900114746148
# 0 26 166188 13.469017016872458
    
# 0 88 1194730 16.839583002017193
# 0 73 74788 16.81442209980211

# with open(r"D:\MolDataSet\moses_mol_tree.pkl", 'rb') as file:
#     Trees=pickle.load(file)
# print(len(Trees))
# for t in Trees[:50]:
#     if t is not None:
#         print(t.smiles)

# voc_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"vocab",'guaca_vocab.txt'))
# with open(voc_path,"r") as file:
#     cset=[x.strip() for x in file.readlines()]

# vocab=set(cset)
# with open(os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"vocab",'vocab_guaca.txt')),"a") as file:
#     for v in vocab:
#         file.write(v+'\n')