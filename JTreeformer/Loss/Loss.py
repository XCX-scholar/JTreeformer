import torch
import torch.nn as nn
import torch.nn.functional as F


def VAE_Loss3(
        batch_data,
        result_node,
        result_edge,
        mean_encoding,
        lnvar_encoding,
        alpha,
        pred_property,
        loss1,
        loss2,
        beta=0,
        gamma=0.2,
        cls_auxiliary=True,
        device=torch.device("cuda:0")
    ):
    x,adj_edge_feature,layers = batch_data['x'],batch_data['adj']
    num_graph,num_node = x.shape[0],x.shape[1]

    '''Calculate CE loss for predicted nodes'''
    num_node_type = result_node[0].shape[-1]
    result_node = result_node[:,:num_node,:].reshape(num_graph*num_node,num_node_type).to(device)
    x = x.reshape(num_graph*num_node)
    mean_CE_node = loss1(result_node,x)
    # print(mean_CE_node)

    '''Calculate CE loss for predicted relationship between prediting and current nodes'''
    relation_type=result_edge.shape[-1]

    mask_adj = adj_edge_feature.masked_fill(
        torch.triu(torch.ones(num_node, num_node, dtype=torch.bool, device=adj_edge_feature.device)).unsqueeze(0), 0)
    flag = 1 - mask_adj.sum(dim=-1).clamp(0, 1)
    flag_adj = torch.cat([flag.unsqueeze(-1), mask_adj], dim=-1)
    father_idx = torch.arange(num_node + 1, device=flag_adj.device).unsqueeze(0).unsqueeze(1).repeat(num_graph,
                                                                                                     num_node, 1)
    cur_father = father_idx[flag_adj.eq(1)].reshape(num_graph, num_node)
    next_father = torch.cat([cur_father[:, 1:], torch.zeros(num_graph, 1, dtype=torch.long, device=cur_father.device)],
                            dim=-1)
    fa = next_father == father_idx[:, 0, 1:]
    bro = cur_father == next_father
    relation = torch.zeros(num_graph, num_node, dtype=torch.long, device=fa.device)
    relation[fa] = 1
    relation[bro] = 2
    relation[~(fa | bro)] = 3
    la=layers[:,:-1]!=layers[:,1:]
    la=torch.cat([la,torch.zeros((layers.shape[0],1),dtype=torch.bool,device=relation.device)],dim=-1)
    relation[la]=4
    relation[flag.eq(1)] = 0
    relation[:, 0] = 4
    '''Calculate CE loss for predicted relationship'''
    relation=relation.reshape(-1)
    result_edge = result_edge[:,:num_node,:].reshape(num_graph*num_node,relation_type).to(device)
    mean_CE_edge = loss2(result_edge,relation)

    # print(mean_CE_edge)

    del mask_adj,flag,flag_adj,father_idx,cur_father,next_father,fa,bro,relation

    '''calculate KL divergence loss for the encoding part.'''
    KL = -0.5*(lnvar_encoding - torch.square(mean_encoding) - torch.exp(lnvar_encoding)+1).mean()
    # print(KL)

    '''calculate property loss for the predicted property through CLS network'''
    if cls_auxiliary:
        property=batch_data['property']
        w_loss=F.mse_loss(pred_property[:,0],property[:,0])
        logp_loss = F.mse_loss(pred_property[:,1], property[:,1])
        tpsa_loss = F.mse_loss(pred_property[:,2], property[:,2])
        property_loss=w_loss+logp_loss+tpsa_loss
        return (mean_CE_node + alpha*mean_CE_edge)/(1+alpha) + beta*KL+gamma*property_loss,mean_CE_node,mean_CE_edge,KL,w_loss,logp_loss,tpsa_loss
    else:
        return (mean_CE_node + alpha*mean_CE_edge)/(1+alpha) + beta*KL,mean_CE_node,mean_CE_edge,KL,0


'''calculate the loss for the DDPM model'''
def DDPM_Loss(noise,predicted_noise,loss_type,device=torch.device("cuda:0")):
    if loss_type == "l1":
        losses = F.l1_loss(predicted_noise, noise, reduction='mean')
    elif loss_type == "l2":
        losses = F.mse_loss(predicted_noise, noise, reduction='mean')
    elif loss_type == "huber":
        losses = F.smooth_l1_loss(predicted_noise,noise, reduction='mean')
    else:
        raise NotImplementedError()

    return losses.to(device)
