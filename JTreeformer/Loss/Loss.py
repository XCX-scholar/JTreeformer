import torch
import torch.nn as nn
import torch.nn.functional as F


# def encoder_loss(
#         x,
#         labels,
#         masked_token_logit,
#         class_logit,
#         masked_idx,
#         loss1,
#         loss2,
#         alpha=1,
#         device=torch.device("cuda:0")
#     ):
#     num_graph, num_node = x.shape[0], x.shape[1]
#     num_node_type = masked_token_logit.shape[-1]
#     masked_token = x
#     masked_token[~masked_idx] = 0
#     masked_token = masked_token.reshape(num_graph*num_node).long().to(device)
#     if loss2 is not None and labels is not None:
#         num_label = labels.shape[-1]
#         labels=labels.reshape(num_graph*num_label)
#         token_loss=loss1(masked_token_logit.reshape(num_graph*num_node,num_node_type),masked_token)
#         cls_loss=alpha*loss2(class_logit.reshape(num_graph*num_label),labels.float())
#         return (token_loss+cls_loss)/2,token_loss,cls_loss
#     else:
#         return loss1(masked_token_logit.reshape(num_graph*num_node,num_node_type),masked_token)
#
# def decoder_loss(
#         batch_data,
#         result_node,
#         result_edge,
#         alpha,
#         loss1,
#         loss2,
#         device=torch.device("cuda:0")
#     ):
#     x,adj_edge_feature = batch_data['x'],batch_data['adj']
#     num_graph,num_node = x.shape[0],x.shape[1]
#     num_node_type = result_node.shape[-1]
#     result_node = result_node[:,:num_node,:].reshape(num_graph*num_node,num_node_type).to(device)
#     x = x[:,:,0].long().to(device).reshape(num_graph*num_node)
#
#     mean_CE_node = loss1(result_node,x)
#     # print(mean_CE_node)
#
#     num_edge_type = result_edge.shape[-1]
#     adj_edge_feature = adj_edge_feature[:,:,:,0].long().to(device)
#     mask = torch.triu(torch.ones((num_node, num_node), dtype=torch.bool, device=device), diagonal=1)
#     adj_edge_feature = adj_edge_feature.masked_fill(mask.unsqueeze(0), 0)
#     adj_edge_feature = adj_edge_feature.reshape(num_graph*num_node*num_node)
#     result_edge = result_edge[:,:num_node,:num_node,:].reshape(num_graph*num_node*num_node,num_edge_type).to(device)
#
#     mean_CE_edge = loss2(result_edge,adj_edge_feature)
#     print(f"mean_CE_node: {mean_CE_node.item()} ,mean_CE_edge: {mean_CE_edge.item()} ")
#
#     # print(mean_CE_edge)
#     # print(KL)
#
#     return 0.5*(mean_CE_node + alpha* mean_CE_edge)

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
    x,adj_edge_feature = batch_data['x'],batch_data['adj']
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
    relation[flag.eq(1)] = 0
    relation[:, 0] = 1
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

def Reg_Loss(
        property,
        pred_property
    ):
    w_loss = F.mse_loss(pred_property[:, 0],property[:, 0])
    logp_loss = F.mse_loss(pred_property[:, 1],property[:, 1])
    tpsa_loss = F.mse_loss(pred_property[:, 2],property[:, 2])
    property_loss = w_loss + logp_loss + tpsa_loss
    return property_loss/3,w_loss.item(),logp_loss.item(),tpsa_loss.item()

def CFS_Map_Loss(
        labels,
        graph_encoding,
        label_encoding,
        component_labels_prob,
        alpha=1,
        device=torch.device("cuda:0")
    ):
    CFS_dim = graph_encoding.shape[-1]
    num_graph = graph_encoding.shape[0]
    mean_CE = (labels*torch.log(component_labels_prob+1e-4) + (1-labels)*torch.log(1-component_labels_prob-1e-4)).mean()

    print()

    var_graph = torch.einsum('ij,ik->jk',graph_encoding,graph_encoding) / num_graph
    # dim:[CFS_dim,CFS_dim]
    var_label = torch.einsum('ijk,ijl->jkl', label_encoding,label_encoding) / num_graph
    # dim:[num_labels,CFS_dim,CFS_dim]
    limit = torch.square(var_graph - torch.eye(CFS_dim,device=device)).mean() + torch.square(var_label - torch.eye(CFS_dim,device=device).unsqueeze(0)).mean()

    return -1*mean_CE + alpha*limit


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
