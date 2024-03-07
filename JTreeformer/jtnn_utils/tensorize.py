import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import jtnn_utils.chemutils as chemutils
from jtnn_utils.mol_tree import *
import json
import torch
import numpy as np


def bfs_ranking(node_list):
    res=[node_list[0]]
    q=[node_list[0]]
    q[0].layer=1
    fa_nids=[q[0].nid]
    while(len(q)>0):
        if len(q[0].neighbors) >0:
            deg=[]
            fa_nids.append(q[0].nid)
            for nei in q[0].neighbors:
                deg.append(len(nei.neighbors))
            idx=np.argsort(np.array(deg))[::-1]
            for i in idx:
                nei=q[0].neighbors[i]
                if nei.nid not in fa_nids:
                    nei.layer=q[0].layer+1
                    q.append(nei)
                    res.append(nei)
        q.pop(0)
    for i in range(len(res)):
        res[i].nid=i+1
    return res

def convert(MolTrees,length,vocab_map):
    n=[]
    a=[]
    p=[]
    l=[]
    h=[]
    for i,tree in enumerate(MolTrees):
        node_ = torch.zeros(length+1, dtype=torch.int16)
        hs_=torch.zeros(length+1,dtype=torch.int8)
        adj_ = torch.zeros((length+1,length+1), dtype=torch.int8)
        property_ = torch.zeros(3, dtype=torch.float)
        layer_ = torch.zeros(length+1, dtype=torch.int8)
        nodes=bfs_ranking(tree.nodes)
        if len(nodes) >0:
            for j,node in enumerate(nodes):
                node_[j]=vocab_map[node.smiles]
                for nei in node.neighbors:
                    adj_[node.nid-1,nei.nid-1]=1
                    adj_[nei.nid-1,node.nid-1]=1
                layer_[j]=node.layer
                hs_[j]=node.hs

            node_[len(tree.nodes)]=vocab_map['stop']
            property_[0]=tree.w/100
            property_[1]=tree.logp
            property_[2]=tree.tpsa/100
            mask_adj = adj_.masked_fill(torch.triu(torch.ones(length+1,length+1, dtype=torch.bool)).unsqueeze(0), 0)
            tmp = mask_adj.sum(dim=-1)
            tmp = tmp[tmp > 1]
            if tmp.numel()>0:
                print(i)
            else:
                n.append(node_)
                a.append(adj_)
                p.append(property_)
                l.append(layer_)
                h.append(hs_)
        if i % 1000==0:
            print(i)



    return torch.stack(n,dim=0),torch.stack(h,dim=0),torch.stack(a,dim=0),torch.stack(p,dim=0),torch.stack(l,dim=0)

if __name__=='__main__':
    import pickle
    save_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"MolDataSet","valid_guaca_mol_tree.pkl"))
    with open(save_path,'rb') as file:
        MolTrees=pickle.load(file)
    voc_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"vocab",'vocab_map_guaca.json'))
    with open(voc_path,'r') as file:
        vocab_map=json.load(file)
    n=len(MolTrees)
    num_nodes=0
    max=0
    min=0
    for tree in MolTrees:
        num_node=len(tree.nodes)
        num_nodes+=num_node
        if num_node>max:
            max=num_node
        if num_node<min:
            min=num_node
    print(min, max, n, num_nodes / n)
    node_,hs_,adj_,pro_,layer_=convert(MolTrees=MolTrees,length=90,valid_num=0,vocab_map=vocab_map)
    adj_=adj_.bool()
    data={}
    data['x']=node_
    data['hs']=hs_
    data["property"]=pro_
    data['adj']=adj_
    data['layer_number']=layer_
    print(node_.shape)
    print(hs_.max(),pro_[:,0].max(),pro_[:,1].max(),pro_[:,2].max())
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"MolDataSet","valid_data_guaca.pkl")),'wb') as file:
        pickle.dump(data,file)

# # 0 40 164664 17.752933245882524
# # 38 449.6430 3.3969 225.9800
# # 13 449.5510 1.9607 188.
