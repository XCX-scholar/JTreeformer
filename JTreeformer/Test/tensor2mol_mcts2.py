from module.JTreeformer import JTreeformer
from jtnn_utils.mol_tree import *
from jtnn_utils.chemutils import *
from jtnn_utils.vocab import *
from rdkit.Chem import Descriptors
import torch
import rdkit.Chem as Chem
import copy, math
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

def bond_inter(smiles1,smiles2):
    ctr_mol=Chem.MolFromSmiles(smiles1)
    nei_mol=Chem.MolFromSmiles(smiles2)
    for b1 in ctr_mol.GetBonds():
        for b2 in nei_mol.GetBonds():
            if ring_bond_equal(b1, b2) or ring_bond_equal(b1, b2, reverse=True):
                return True

    return False

def have_charge(fa_node,chi_node):
    cap=fa_node.hs
    for nei in fa_node.neighbors+[chi_node]:
        if len(nei.smiles)==1:
            continue
        if len(nei.smiles)==2:
            cap-=1
        if len(nei.smiles)==3:
            if nei.smiles[1]=='=':
                cap-=2
            if nei.smiles[1]=='#':
                cap-=3
        if len(nei.smiles)>3:
            if nei.smiles.__contains__('='):
                cap-=3
            elif nei.smiles.__contains__('#'):
                cap-=4
            else:
                cap-=2
    return (cap>=0)

def have_slots(fa_slots,ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = list(zip(*matches))
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True

def can_assemble(node_x, node_y):
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = (len(nei.neighbors) <= 1)
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands,aroma_scores = enum_assemble(node_x, neighbors,bl=False,cs=True)
    return len(cands) > 0# and sum(aroma_scores) >= 0



def dfs_random(cur_mol,global_amap,fa_amap,cur_node,fa_node,stack,length,check_aroma):
    while(len(stack)>0):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node,neighbors,prev_nodes,cur_amap,cs=True,bl=False)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            child_ord,_,prev_amap=stack[-1]
            stack.pop(-1)
            if len(stack) > 0:
                _, prev_node,_= stack[-1]
                while (len(stack) > 0 and child_ord + 1 >= len(prev_node.children)):
                    child_ord,_,prev_amap= stack[-1]
                    stack.pop(-1)
                    if len(stack) > 0:
                        _,prev_node,_=stack[-1]
                if len(stack) > 0:
                    new_node = prev_node.children[child_ord + 1]
                    stack.append((child_ord + 1, new_node,prev_amap))
                else:
                    return cur_mol,length
            else:
                return cur_mol,length
            fa_node = prev_node
            cur_node = new_node
            fa_amap = prev_amap
            continue
        if len(cands) > 1:
            scores = torch.Tensor(aroma_score)
        else:
            scores = torch.Tensor([1.0])
        _, cand_idx = torch.sort(scores, descending=True)

        cand_smiles,cand_amap=list(zip(*cands))

        pre_mol=cur_mol
        backup_mol = Chem.RWMol(cur_mol)
        next_idx=np.random.randint(len(cands))
        cur_mol = Chem.RWMol(backup_mol)
        pred_amap=cand_amap[next_idx]
        new_global_amap=copy.deepcopy(global_amap)

        for nei_id, ctr_atom, nei_atom in pred_amap:
            if nei_id == fa_nid:
                continue
            new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

        cur_mol = attach_mols(cur_mol,children,[],new_global_amap)  # father is already attached
        new_mol = cur_mol.GetMol()
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
        if new_mol is None:
            child_ord,_,prev_amap=stack[-1]
            stack.pop(-1)
            if len(stack) > 0:
                _, prev_node,_= stack[-1]
                while (len(stack) > 0 and child_ord + 1 >= len(prev_node.children)):
                    child_ord, _,prev_amap= stack[-1]
                    stack.pop(-1)
                    if len(stack) > 0:
                        _, prev_node,_= stack[-1]
                if len(stack) > 0:
                    new_node = prev_node.children[child_ord + 1]
                    stack.append((child_ord + 1,new_node,prev_amap))
                else:
                    return cur_mol, length
            else:
                return cur_mol, length
            fa_node = prev_node
            cur_node = new_node
            cur_mol=pre_mol
            fa_amap=prev_amap
            continue
        length+=len(cur_node.children)
        if not cur_node.is_leaf:
            new_node = cur_node.children[0]
            stack.append((0, new_node,pred_amap))
            prev_node=cur_node
            fa_amap = pred_amap
        else:
            child_ord,_,prev_amap= stack[-1]
            stack.pop(-1)
            if len(stack) > 0:
                _, prev_node,_= stack[-1]
                while (len(stack) > 0 and child_ord + 1 >= len(prev_node.children)):
                    child_ord,_,prev_amap= stack[-1]
                    stack.pop(-1)
                    if len(stack) > 0:
                        _, prev_node,_= stack[-1]
                if len(stack) > 0:
                    new_node=prev_node.children[child_ord + 1]
                    stack.append((child_ord + 1,new_node,prev_amap))
                    fa_amap=prev_amap
                else:
                    return cur_mol,length
            else:
                return cur_mol,length
        fa_node=prev_node
        cur_node=new_node
        global_amap=new_global_amap


class MCTNode():
    def __init__(self,cur_mol,global_amap,fa_amap,cur_node,fa_node,stack,aroma_score):
        self.cur_mol=cur_mol
        self.global_amap=global_amap
        self.fa_amap=fa_amap
        self.cur_node=cur_node
        self.fa_node=fa_node
        self.stack=stack
        self.score=aroma_score
        self.UCT=0
        self.length=1
        self.is_leaf=True
        self.is_end=False
        self.exp_num=0
        self.children=[]
        self.father=None

    def del_node(self):
        del self.cur_mol
        del self.global_amap
        del self.fa_amap
        del self.cur_node
        del self.fa_node
        del self.stack
        del self.score
        del self.UCT
        del self.length
        del self.is_leaf
        del self.exp_num
        del self.children
        del self.father



class MCT():
    def __init__(self,all_nodes,cur_mol,global_amap,fa_amap,cur_node,fa_node,stack,check_aroma):
        self.all_nodes=all_nodes
        self.root=MCTNode(cur_mol,global_amap,fa_amap,cur_node,fa_node,stack,0)
        self.root.exp_num=1
        self.exp_num=1
        self.node_list=[self.root]
        self.check_aroma=check_aroma

    def select(self,node:MCTNode):
        selected=node.children[0]
        max_UCT=selected.UCT
        no=0
        for i,c in enumerate(node.children):
            if c.exp_num==0:
                selected=c
                break
            if c.UCT>max_UCT:
                max_UCT=c.UCT
                selected=c
        return selected

    def expansion(self,node:MCTNode):
        node.is_leaf=False
        fa_nid=node.fa_node.nid if node.fa_node is not None else -1
        prev_nodes = [node.fa_node] if node.fa_node is not None else []

        children = [nei for nei in node.cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors
        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in node.fa_amap if nid == node.cur_node.nid]
        cands,aroma_score = enum_assemble(node.cur_node,neighbors,prev_nodes,cur_amap,cs=False,bl=False)

        if len(cands) == 0 or (sum(aroma_score) < 0 and self.check_aroma):
            node.is_leaf=True
            node.is_end=True
            return 0

        cand_smiles,cand_amap=list(zip(*cands))

        backup_mol = Chem.RWMol(node.cur_mol)
        for i in range(len(cands)):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[i]
            a_s=aroma_score[i]
            new_global_amap = copy.deepcopy(node.global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[node.cur_node.nid][ctr_atom]
            cur_mol=attach_mols(cur_mol,children,[],new_global_amap)  # father is already attached
            new_mol=cur_mol.GetMol()
            new_mol=Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
            if new_mol is None:
                continue
            new_stack=copy.deepcopy(node.stack)
            if not node.cur_node.is_leaf:
                new_node=node.cur_node.children[0]
                new_stack.append((0,new_node,pred_amap))
                new_mctnode=MCTNode(cur_mol,new_global_amap,pred_amap,new_node,node.cur_node,new_stack,a_s)
                new_mctnode.father=node
                new_mctnode.length=node.length+len(node.cur_node.children)
                node.children.append(new_mctnode)
                self.node_list.append(new_mctnode)
                # print('can:',i,'1')
            else:
                child_ord,_,fa_amap=new_stack[-1]
                new_stack.pop(-1)
                if len(new_stack)>0:
                    _,prev_node,_=new_stack[-1]
                    while(len(new_stack)>0 and child_ord+1>=len(prev_node.children)):
                        child_ord,_,fa_amap=new_stack[-1]
                        new_stack.pop(-1)
                        if len(new_stack) > 0:
                            _,prev_node,_ = new_stack[-1]
                    if len(new_stack)>0:
                        new_stack.append((child_ord+1,prev_node.children[child_ord+1],fa_amap))
                        new_mctnode=MCTNode(cur_mol,new_global_amap,pred_amap,prev_node.children[child_ord+1],prev_node,new_stack,a_s)
                        new_mctnode.father=node
                        new_mctnode.length=node.length+len(node.cur_node.children)
                        node.children.append(new_mctnode)
                        self.node_list.append(new_mctnode)
                        # print('can:',i,'2')
                    else:
                        new_mctnode=MCTNode(cur_mol,new_global_amap,pred_amap,None,None,new_stack,a_s)
                        new_mctnode.father=node
                        new_mctnode.length=node.length+len(node.cur_node.children)
                        node.children.append(new_mctnode)
                        new_mctnode.is_end = True
                        self.node_list.append(new_mctnode)
                        # print('can:',i,'3')
                else:
                    new_mctnode = MCTNode(cur_mol, new_global_amap, pred_amap, None, None, new_stack, a_s)
                    new_mctnode.father = node
                    new_mctnode.length = node.length + len(node.cur_node.children)
                    node.children.append(new_mctnode)
                    new_mctnode.is_end = True
                    new_mctnode.is_leaf = True
                    self.node_list.append(new_mctnode)
                    # print('can:',i,'4')
        if len(node.children)==0:
            node.is_end=True
            node.is_leaf = True
        if len(node.children)==1:
            return 1
        else:
            return 0

    def likelihood_score(self,length,w,logp,tpsa,tgtw=None,tgtlogp=None,tgttpsa=None):
        term=1
        likelihood_score=5000*(length/len(self.all_nodes))
        if tgtw is not None:
            tmp=0 if w > tgtw else (tgtw-w)**2
            likelihood_score+=1000*math.exp(-tmp)
            term+=1
        if tgtlogp is not None:
            likelihood_score+=1000*math.exp(-(tgtlogp-logp)**2)
            term+=1
        if tgttpsa is not None:
            likelihood_score+=1000*math.exp(-(tgttpsa-tpsa)**2)
            term+=1
        return likelihood_score/term

    def simulate(self,node:MCTNode,sim_times=2):
        score=0
        if not node.is_end:
            for i in range(sim_times):
                cur_mol,length=dfs_random(
                    copy.deepcopy(node.cur_mol),
                    copy.deepcopy(node.global_amap),
                    copy.deepcopy(node.fa_amap),
                    copy.deepcopy(node.cur_node),
                    copy.deepcopy(node.fa_node),
                    copy.deepcopy(node.stack),
                    copy.deepcopy(node.length),
                    check_aroma=self.check_aroma
                )
                if cur_mol is not None:
                    try:
                        tmp_mol=cur_mol.GetMol()
                        set_atommap(tmp_mol)
                        tmp_mol= Chem.MolFromSmiles(Chem.MolToSmiles(tmp_mol))
                        w = Descriptors.MolWt(tmp_mol)
                        logp = Descriptors.MolLogP(tmp_mol)
                        tpsa = Descriptors.TPSA(tmp_mol)
                    except:
                        w=-100
                        logp=-100
                        tpsa=-100
                    score+=self.likelihood_score(length,w=w,logp=logp,tpsa=tpsa)
                else:
                    score-=5000
            node.score+=score/sim_times
        else:
            cur_mol,length = node.cur_mol,node.length
            if cur_mol is not None:
                try:
                    tmp_mol = cur_mol.GetMol()
                    set_atommap(tmp_mol)
                    tmp_mol = Chem.MolFromSmiles(Chem.MolToSmiles(tmp_mol))
                    w = Descriptors.MolWt(tmp_mol)
                    logp = Descriptors.MolLogP(tmp_mol)
                    tpsa = Descriptors.TPSA(tmp_mol)
                except:
                    w = -100
                    logp = -100
                    tpsa = -100
                score += self.likelihood_score(length, w=w, logp=logp, tpsa=tpsa)
            else:
                score -= 5000


    def bp(self,node:MCTNode):
        cur_node=node
        cur_score=cur_node.score
        while cur_node.father is not None:
            cur_node.exp_num+=1
            if cur_node.is_end:
                cur_node.score=cur_node.score/cur_node.exp_num*(cur_node.exp_num+1)
            cur_node.UCT=cur_node.score+math.sqrt(2)*math.sqrt(math.log(self.exp_num)/cur_node.exp_num)
            cur_node=cur_node.father
            cur_node.score+=cur_score
            self.exp_num += 1

    def Decision(self,times):
        cur_node = self.root
        for i in range(times):
            if cur_node.is_leaf:
                if cur_node.exp_num==0:
                    self.simulate(cur_node)
                    self.bp(cur_node)
                    cur_node=self.root
                else:
                    if not cur_node.is_end:
                        state=self.expansion(cur_node)
                        if i==0 and state==1:
                            break
                    else:
                        self.bp(cur_node)
                        cur_node = self.root

            else:
                if cur_node.is_end:
                    cur_node=self.root
                else:
                    next_node=self.select(cur_node)
                    cur_node=next_node

        if len(self.root.children)>0:
            selected=self.root.children[0]
            max_UCT=selected.UCT
            for i,c in enumerate(self.root.children):
                if c.UCT>max_UCT:
                    max_UCT=c.UCT
                    selected=c
            cur_mol,global_amap,fa_amap,fa_node,cur_node,stack=selected.cur_mol, selected.global_amap, selected.fa_amap,selected.fa_node,selected.cur_node,selected.stack
            for n in self.node_list:
                n.del_node()
            return cur_mol,global_amap,fa_amap,cur_node,fa_node,stack
        else:
            return None,None,None,None,None,None


def dfs_mcts(all_nodes,cur_mol,global_amap,fa_amap,cur_node,fa_node,stack,check_aroma):
    cnt=1
    aaa=0
    if len(all_nodes)==1:
        return cur_mol,cur_mol
    while(len(stack)>0):
        _,tmp,_=stack[-1]
        cnt+=len(tmp.children)
        pre_mol=cur_mol
        mct = MCT(all_nodes, copy.deepcopy(cur_mol), global_amap, fa_amap, cur_node, fa_node, stack,check_aroma)
        cur_mol,global_amap,fa_amap,cur_node,fa_node,stack = mct.Decision(8)
        if cur_mol is None:
            print('assm fail,cur node:',cnt)
            return None,pre_mol
        aaa+=1
    print('assm complete,cur node:',cnt)
    return cur_mol,pre_mol


def PredictNextNode(
        model:JTreeformer,
        latent_encoding:torch.Tensor,
        node_list:torch.Tensor,
        hs:torch.Tensor,
        adj:torch.Tensor,
        layer_number:torch.Tensor,
        use_kv_cache=False
    ):
    assert latent_encoding.shape[0]==1
    num_node=node_list.shape[1]
    num_graph=1
    with torch.no_grad():
        attn_mask = torch.triu(torch.ones(num_node+1,num_node+1),diagonal=1).bool().to(model.device)
        padding_mask = (node_list[:, :]).eq(0).to(model.device)
        padding_mask_cls = torch.zeros(num_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1).to(model.device)

        masked_adj = adj.masked_fill(
            torch.triu(torch.ones(num_node, num_node, dtype=torch.bool, device=adj.device)),
            0.0
        )
        flag = 1 - masked_adj.sum(dim=-1).clamp(0, 1)
        flag_adj = torch.cat([flag.unsqueeze(-1), masked_adj], dim=-1)
        father_idx = torch.arange(num_node + 1, device=flag_adj.device).unsqueeze(0).unsqueeze(1).repeat(num_graph,
                                                                                                         num_node, 1)

        father_position = father_idx[flag_adj.eq(1)].reshape(num_graph,num_node)
        bro_ord = torch.cumsum(masked_adj, dim=-2)
        bro_ord = torch.gather(bro_ord, dim=-1, index=(father_position - 1).clamp(min=0).unsqueeze(-1))
        bro_ord[flag.eq(1)] = 0
        bro_ord=bro_ord.long().reshape(num_graph,num_node)
        adj_with_cls = torch.cat([torch.ones(num_graph, num_node, 1, device=model.device, dtype=adj.dtype), masked_adj],
                                 dim=2)
        adj_with_cls = torch.cat(
            [torch.zeros(num_graph, 1, num_node + 1, device=model.device, dtype=adj.dtype), adj_with_cls], dim=1)

        degree = masked_adj.sum(dim=-1)
        degree_with_cls = torch.cat([torch.zeros(num_graph, 1, device=model.device), degree], dim=1)
        degree_diag = torch.diag_embed(torch.sqrt(1 / (degree_with_cls + 1)))

        T = torch.diag_embed(degree_with_cls + 1) - adj_with_cls
        T = torch.bmm(degree_diag, T)
        T = torch.bmm(T, degree_diag)

        decoder_bias = model.decoder_bias(adj)
        # print("decoder bias finished")

        decoder_node_feature = model.decoder_node_feature(
            x=node_list,
            hs=hs,
            father_position=father_position,
            brother_order=bro_ord,
            layer_number=layer_number.long()
        )
        if not use_kv_cache:
            decoder_node_feature = torch.cat(
                [decoder_node_feature,latent_encoding.unsqueeze(1).repeat(1, num_node + 1, 1)],
                dim=-1)
            decoder_node_feature = model.fusion_attn(x=decoder_node_feature, attn_mask=attn_mask, padding_mask=padding_mask)

            result_node, node_decoding = model.decoder(x=decoder_node_feature, T=T, thetas=model.thetas, attn_mask=attn_mask,
                                                      padding_mask=padding_mask, attn_bias=decoder_bias)
            return result_node[:,-1,:],node_decoding
        else:
            decoder_node_feature = torch.cat(
                [decoder_node_feature, latent_encoding.unsqueeze(1).repeat(1, num_node + 1, 1)],
                dim=-1)
            decoder_node_feature = model.fusion_attn(x=decoder_node_feature, attn_mask=attn_mask,
                                                     padding_mask=padding_mask)

            result_node, node_decoding = model.decoder(x=decoder_node_feature[:,-2:-1,:], T=T, thetas=model.thetas,
                                                       attn_mask=attn_mask,
                                                       padding_mask=None, attn_bias=decoder_bias,use_kv_cache=use_kv_cache)
            return result_node, node_decoding


def PredictRelation(
        model:JTreeformer,
        node_decoding:torch.Tensor,
        node_list:torch.Tensor,
    ):
    assert node_decoding.shape[0]==1
    with torch.no_grad():
        num_graph,num_node = node_list.shape[:2]
        attn_mask = torch.triu(torch.ones(num_node,num_node),diagonal=1).bool().to(model.device)
        padding_mask = (node_list[:, :]).eq(0).to(model.device)
        # dim[num_graph,num_node,1]
        decoder_adj_feature = model.adj_feature(node_list)
        decoder_adj_feature = model.adj_cross_attn(
            x=node_decoding, key=decoder_adj_feature,
            value=decoder_adj_feature,
            attn_mask=attn_mask,
            padding_mask=padding_mask
        )
        relation_logit = model.relation_logit_proj(decoder_adj_feature)

    return relation_logit[:,-1,1:]

def Tensor2MolTree(model:JTreeformer,
                   latent_encoding,
                   vocab:Vocab,
                   max_decode_step,
                   prob_decode_node=False,
                   prob_decode_edge=False,
                   device="cuda:0",
                   use_black_list=False,
                   filter_sington=False,
                   bond_limit=False,
                   checking_charge=False,
                   use_kv_cache=False
                   ):
        assert latent_encoding.shape[0] == 1

        list = []
        layer_first=[]
        num_node=0
        node_list=torch.zeros((1,num_node), dtype=torch.long, device=torch.device(device))
        hs=torch.zeros((1, num_node), dtype=torch.long, device=torch.device(device))
        adj = torch.full((1,num_node,num_node),fill_value=1,dtype=torch.long,device=torch.device(device))
        layer_number=torch.zeros((1,num_node), dtype=torch.long, device=torch.device(device))
        sing=[i+1 for i in range(len(vocab.vocab)-1) if len(vocab.vocab[i+1])==1]
        # ring=[i+1 for i in range(len(vocab.vocab)-1) if len(vocab.vocab[i+1])>3]
        if use_black_list:
            black_list=[i+1 for i in range(len(vocab.vocab)-1) if (len(vocab.vocab[i+1])==2 and vocab.vocab[i+1].__contains__('C') and vocab.vocab[i+1]!='CC')]+[vocab.vmap['C']]

        next_node_logit,node_decoding=PredictNextNode(model,latent_encoding,node_list,hs,adj,layer_number,use_kv_cache=use_kv_cache)
        prob_node = torch.softmax(next_node_logit, dim=-1)
        if prob_decode_node:
            sort_wid = torch.multinomial(prob_node[:,:vocab.vmap['stop']],1)
        else:
            _, sort_wid = torch.sort(next_node_logit[:,:vocab.vmap['stop']], dim=1, descending=True)
            sort_wid = sort_wid[:, :1]
        # root_wid = ring[sort_wid.item()]
        root_wid=sort_wid.item()

        root = MolTreeNode(vocab.get_smiles(root_wid))
        root.fa=None
        root.wid = root_wid
        root.idx = 0
        root.children=[]
        list.append( (root,vocab.get_slots(root.wid)) )
        node_list=torch.cat([node_list,torch.full((1,1),fill_value=root_wid,device=node_list.device,dtype=torch.long)],dim=-1)
        hs=torch.cat([hs,torch.full((1,1),root.hs,dtype=torch.long,device=hs.device)], dim=-1)
        adj = torch.zeros((1,1,1),dtype=torch.long, device=adj.device)
        layer_number=torch.cat([layer_number,torch.ones(1,1,dtype=torch.long, device=adj.device)],dim=-1)
        all_nodes = [root]
        num_node+=1
        layer_first.append(0)

        stop=False
        with torch.no_grad():
            while(not stop):
                next_node_logit,node_decoding = PredictNextNode(model,latent_encoding,node_list,hs,adj,layer_number,use_kv_cache=use_kv_cache)
                next_node_logit=next_node_logit[:,:vocab.vmap['stop']+1]
                # print(next_node_logit.shape,vocab.vmap['stop'])
                if prob_decode_node:
                    prob_node = torch.softmax(next_node_logit, dim=-1)
                    while (True):
                        sort_wid = torch.multinomial(prob_node,15)
                        if sort_wid[sort_wid == 0].numel() == 0:
                            break
                else:
                    _,sort_wid = torch.sort(next_node_logit, dim=1, descending=True)
                    sort_wid = sort_wid[:,:15]

                next_wid = None
                for i in range(15):
                    wid=sort_wid[:,i].item()
                    if wid==vocab.vmap['stop'] or wid==0:
                        break
                    if use_black_list:
                        if root_wid in black_list+[vocab.vmap['CC']] and (wid in black_list and num_node<2):
                            continue
                    if filter_sington:
                        if wid in sing:
                            continue
                    cand_node_list=torch.cat([node_list,sort_wid[:,i].unsqueeze(-1)],dim=-1)
                    relation_logit = PredictRelation(model, node_decoding,cand_node_list)
                    if prob_decode_edge:
                        prob_relation = torch.softmax(relation_logit, dim=-1)
                        sort_relation = torch.multinomial(prob_relation,4,replacement=True)
                    else:
                        _,sort_relation = torch.sort(relation_logit, dim=-1, descending=True)
                    # print(wid)
                    slots=vocab.get_slots(wid)
                    node_y=MolTreeNode(vocab.get_smiles(wid))
                    fa_id=None
                    can_assm=False
                    for j in range(4):
                        if num_node==1:
                            fa_id=0
                        else:
                            relation=sort_relation[:,j].item()
                            if relation==0:
                                fa_id=num_node-1
                            elif relation==1:
                                fa_id=all_nodes[num_node-1].fa.idx
                            elif relation==2:
                                fa_id = all_nodes[num_node - 1].fa.idx+1
                            else:
                                fa_id=layer_first[-1]
                        node_x,fa_slot=list[fa_id]
                        if checking_charge:
                            if not have_charge(node_x,node_y) or len(node_x.neighbors)+1>=8:
                                continue
                        else:
                            if len(node_x.neighbors) + 1 >= 8:
                                continue
                        if node_x.wid in sing:
                            continue
                        cur_s=vocab.get_smiles(wid)
                        if bond_limit:
                            if len(node_x.smiles)==2:
                                ctn=False
                                neis=node_x.neighbors+[] if node_x.fa is None else node_x.neighbors+node_x.fa.children
                                for nei in neis:
                                    fa_s=nei.smiles
                                    if ((len(fa_s)==3 and len(cur_s)>3) or (len(fa_s)>3 and len(cur_s)==3)) and not bond_inter(fa_s,cur_s):
                                        ctn=True
                                        break
                                if ctn:
                                    continue
                        if have_slots(fa_slot,slots) and can_assemble(node_x,node_y):
                            can_assm=True
                            next_wid = wid
                            next_slots = slots
                            next_node=sort_wid[:,i].reshape(1,1)
                            break
                    if can_assm:
                        break
                if next_wid is None:
                    break
                if relation==3:
                    layer_first.append(num_node)
                node_y = MolTreeNode(vocab.get_smiles(next_wid))
                node_y.wid = next_wid
                node_y.idx = len(all_nodes)
                node_y.neighbors.append(node_x)
                node_y.fa=node_x
                node_y.children=[]
                node_x.neighbors.append(node_y)
                node_x.children.append(node_y)
                list.append((node_y,next_slots))
                all_nodes.append(node_y)
                adj=torch.cat([adj,torch.zeros((1,1,num_node),dtype=torch.long,device=adj.device)],dim=1)
                adj[:,-1,fa_id]=1
                adj=torch.cat([adj,torch.zeros((1,num_node+1,1),dtype=torch.long,device=adj.device)],dim=-1)
                adj[:,fa_id,-1]=1
                node_list=torch.cat([node_list,next_node],dim=1)
                hs=torch.cat([hs,torch.full((1,1),node_y.hs,dtype=torch.long,device=hs.device)], dim=1)
                layer_number=torch.cat([layer_number,(layer_number[:,fa_id]+1).unsqueeze(-1)],dim=1)

                num_node+=1
                if num_node>=max_decode_step:
                    stop=True
            # print(adj)
            # print(node_list)
            # print(num_node)

            for i,node in enumerate(all_nodes):
                node.nid=i+1

        model.decoder._reset_cache()
        return root, all_nodes


def MolTree2Mol(pred_root:MolTreeNode,pred_nodes:list):
    if len(pred_nodes) == 0:
        return None
    elif len(pred_nodes) == 1:
        return pred_root.smiles

    max_len=0
    max_wid=-1
    # Mark nid & is_leaf & atommap
    for i, node in enumerate(pred_nodes):
        node.nid = i + 1
        node.is_leaf = (len(node.neighbors) == 1)
        if len(node.neighbors) > 1:
            set_atommap(node.mol, node.nid)
        if len(node.neighbors) > max_len:
            max_len=len(node.neighbors)
            max_wid=node.wid
    # print(max_wid)
    pred_root.is_leaf= (len(pred_nodes)==1)

    cur_mol = copy_edit_mol(pred_root.mol)
    global_amap = [{}] + [{} for node in pred_nodes]
    global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
    # print('root is leaf:',pred_root.is_leaf)

    cur_mol,_=dfs_mcts(pred_nodes,cur_mol,global_amap,[],pred_root,None,stack=[(None,pred_root,[])],check_aroma=True)
    if cur_mol is None:
        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
        cur_mol, pre_mol = dfs_mcts(pred_nodes, cur_mol, global_amap, [], pred_root, None, stack=[(None,pred_root,[])],check_aroma=False)
    if cur_mol is None: cur_mol = pre_mol

    if cur_mol is None:
        return None

    cur_mol = cur_mol.GetMol()
    set_atommap(cur_mol)
    cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
    return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
