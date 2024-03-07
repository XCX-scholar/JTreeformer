import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jtnn_utils.mol_tree import MolTree
from jtnn_utils import tensorize
from jtnn_utils.vocab import Vocab
from module.JTreeformer import JTreeformer
from Test.tensor2mol_mcts2 import *
import numpy as np
import pickle
import torch.nn as nn
from tqdm import tqdm

def test_1(smiles_list,vocab_path,model_path,max_decode_step=30,prob_decode_node=False,prob_decode_edge=False,device='cpu'):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    with open(vocab_path, 'r') as file:
        frags=[x.strip() for x in file.readlines()]
    vocab=Vocab(frags)
    model = JTreeformer(
        num_layers_encoder=12,
        num_layers_decoder=12,
        hidden_dim_encoder=512,
        expand_dim_encoder=1024,
        hidden_dim_decoder=768,
        expand_dim_decoder=1536,
        latent_space_dim=768,
        num_head_encoder=16,
        num_head_decoder=16,
        num_node_type=710,
        max_hs=50,
        max_degree=20,
        max_layer_num=50,
        max_brother_num=20,
        dropout=True,
        dropout_rate=0.1,
        device=device,
    )
    model.encoder.add_module('property_proj', nn.Linear(model.encoder.hidden_dim, 4).to(
        torch.device(device)))
    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
    model.encoder.dropout=False
    model.decoder.dropout=False
    vmap = {x: i+1 for i, x in enumerate(frags)}
    vmap['padding']=0
    vmap['stop']=len(vmap)
    batch_data={}
    for smiles in smiles_list:
        mol_tree = MolTree(smiles)
        n,h,a,p,l=tensorize.convert([mol_tree],40,vmap)
        batch_data['x']=n
        batch_data['hs']=h.long()
        batch_data['adj']=a.long()
        batch_data['property']=p
        batch_data['layer_number']=l.long()
        mean_encoding,_,_ = model(batch_data, encoding_only=True)
        latent_encoding=mean_encoding + torch.randn((5,model.decoder.hidden_dim),device=torch.device(device))
        for i in tqdm(range(5),total=5):
            e=latent_encoding[i].unsqueeze(0)
            pred_root, pred_nodes=Tensor2MolTree(model,
                    e,
                    vocab,
                    max_decode_step,
                    prob_decode_node,
                    prob_decode_edge,
                    device)
            if len(pred_nodes)>0:
                new_smiles=MolTree2Mol(pred_root,pred_nodes)
            else:
                new_smiles="None"
            print(len(pred_nodes), new_smiles)

def test_2(smiles1,smiles2,vocab_path,model_path,max_decode_step=30,prob_decode_node=False,prob_decode_edge=False,device='cpu'):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    with open(vocab_path, 'r') as file:
        frags=[x.strip() for x in file.readlines()]
    vocab=Vocab(frags)
    model = JTreeformer(
        num_layers_encoder=12,
        num_layers_decoder=12,
        hidden_dim_encoder=512,
        expand_dim_encoder=1024,
        hidden_dim_decoder=768,
        expand_dim_decoder=1536,
        latent_space_dim=768,
        num_head_encoder=16,
        num_head_decoder=16,
        num_node_type=710,
        max_hs=50,
        max_degree=20,
        max_layer_num=50,
        max_brother_num=20,
        dropout=True,
        dropout_rate=0.1,
        device=device,
    )
    model.encoder.add_module('property_proj', nn.Linear(model.encoder.hidden_dim, 4).to(
        torch.device(device)))
    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
    model.encoder.dropout=False
    model.decoder.dropout=False
    vmap = {x: i+1 for i, x in enumerate(frags)}
    vmap['padding']=0
    vmap['stop']=len(vmap)
    batch_data={}
    mol_tree = MolTree(smiles1)
    n,h,a,p,l=tensorize.convert([mol_tree],40,vmap)
    batch_data['x']=n
    batch_data['hs']=h.long()
    batch_data['adj']=a.long()
    batch_data['property']=p
    batch_data['layer_number']=l.long()
    mean_encoding1,_,_ = model(batch_data, encoding_only=True)

    mol_tree = MolTree(smiles2)
    n,h,a,p,l=tensorize.convert([mol_tree],40,vmap)
    batch_data['x']=n
    batch_data['hs']=h.long()
    batch_data['adj']=a.long()
    batch_data['property']=p
    batch_data['layer_number']=l.long()
    mean_encoding2,_,_ = model(batch_data, encoding_only=True)
    for i in tqdm(range(5),total=5):
        if i==0:
            pass
        e=mean_encoding1* i / 5 + mean_encoding2 * (5 - i) / 5
        print(e.shape)
        pred_root, pred_nodes=Tensor2MolTree(model,
                e.unsqueeze(0),
                vocab,
                max_decode_step,
                prob_decode_node,
                prob_decode_edge,
                device)
        if len(pred_nodes)>0:
            new_smiles=MolTree2Mol(pred_root,pred_nodes)
        else:
            new_smiles="None"
        print(len(pred_nodes), new_smiles)

if '__main__'==__name__:
    model_path = os.path.join(os.path.abspath(''),'vae',"vae_model_moses3_epoch4.pth")
    smiles_list = ['CCCCCCCCCCCCCCC','CCCCCCCOCCCCCCCC']
    test_1(smiles_list,os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','vocab','vocab_moses.txt'),model_path)
    # test_2(smiles_list[0],smiles_list[1],os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','vocab','vocab_moses.txt'),model_path)