import torch
import torch.nn as nn
import math

def position_encoding(position,hidden_dim):
    assert hidden_dim==(hidden_dim//2)*2
    pe=torch.zeros(*(position.shape),hidden_dim,device=position.device)
    factor=torch.exp(torch.arange(0,hidden_dim,2,device=position.device)*(-math.log(10000)/hidden_dim)).unsqueeze(0).unsqueeze(1)

    pe[:,:,0::2]=torch.sin(position.unsqueeze(-1)*factor)
    pe[:,:,1::2]=torch.cos(position.unsqueeze(-1)*factor)
    return pe

class NodeFeature(nn.Module):
    def __init__(
            self,
            num_node_type,
            max_hs,
            max_layer_num,
            max_brother_num,
            hidden_dim,
            max_degree=0,
            decoder=False,
            device="cuda:0",
            feature_test=False
    ):

        super(NodeFeature,self).__init__()
        self.num_node_type = num_node_type
        self.device=torch.device(device)
        self.hidden_dim=hidden_dim
        self.node_encoding = nn.Embedding(num_node_type + 3, hidden_dim, padding_idx=0).to(self.device)
        self.feature_test = feature_test
        if not feature_test:
            self.hs_encoding=nn.Embedding(max_hs+1,hidden_dim,padding_idx=0).to(self.device)
            self.layer_num_encoding = nn.Embedding(max_layer_num+1, hidden_dim, padding_idx=0).to(self.device)
            self.bro_ord_encoding=nn.Embedding(max_brother_num+1, hidden_dim, padding_idx=0).to(self.device)
        self.decoder = decoder
        self.virtual_token_encoding = nn.Embedding(1, hidden_dim).to(self.device) 
        if not decoder:
            self.degree_encoding = nn.Embedding(max_degree + 1, hidden_dim, padding_idx=0).to(self.device)


    def forward(self,x,hs,layer_number,father_position,brother_order,degree=None,T=None):
        x = x.long().to(self.device)
        num_graph,num_node = x.size()[:2]
        node_feature = self.node_encoding(x)
        if not self.feature_test:
            slot_feature=self.hs_encoding(hs)
            node_feature+=slot_feature
            layer_num_feature=self.layer_num_encoding(layer_number)
            node_feature+=layer_num_feature
            pe=position_encoding(father_position,self.hidden_dim)
            node_feature+=pe
            bro_ord_feature=self.bro_ord_encoding(brother_order)
            node_feature+=bro_ord_feature

        if T is not None:
            node_feature=torch.bmm(T,node_feature)
        if not self.decoder:
            degree_feature = self.degree_encoding(degree)
            node_feature = node_feature + degree_feature
        virtual_token_feature = self.virtual_token_encoding.weight.unsqueeze(0).repeat(num_graph,1,1)
        node_feature = torch.cat([virtual_token_feature,node_feature],dim=1)
        return node_feature

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_head,
            k_dim,
            v_dim,
            q_dim,
            require_bias=False,
            self_attention=True,
            dropout=False,
            dropout_rate=0.5,
            device="cuda:0"
    ):

        super(MultiHeadAttention,self).__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.k_proj = nn.Linear(self.k_dim,hidden_dim, bias=require_bias).to(self.device)
        self.q_proj = nn.Linear(self.q_dim,hidden_dim, bias=require_bias).to(self.device)
        self.v_proj = nn.Linear(self.v_dim,hidden_dim, bias=require_bias).to(self.device)
        self.dropout = dropout
        if self.dropout:
            self.Dropout = nn.Dropout(p=dropout_rate)
        # self.reset_parameters()

    def reset_parameters(self):
        if self.k_dim == self.v_dim and self.k_dim == self.q_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain= 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_proj.weight, gain= 2 ** -0.5)
            nn.init.xavier_uniform_(self.q_proj.weight, gain= 2 ** -0.5)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

    def forward(self,x,attn_bias=None,attn_mask=None,padding_mask=None,key=None,value=None):
        num_graph,tgt_node,hidden_dim = x.size()
        if self.self_attention == False:
            _,src_node,_ = key.size()
            q = self.q_proj(x)
            k = self.k_proj(key)
            v = self.v_proj(value)
            q *= self.scaling
        else:
            src_node = tgt_node
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            q *= self.scaling
        q = q.reshape(num_graph,tgt_node,self.num_head,self.head_dim).permute(0,2,1,3).reshape(
            num_graph*self.num_head,
            tgt_node,
            self.head_dim)
        k = k.reshape(num_graph, src_node, self.num_head, self.head_dim).permute(0, 2, 1, 3).reshape(
            num_graph * self.num_head,
            src_node,
            self.head_dim)
        v = v.reshape(num_graph, src_node, self.num_head, self.head_dim).permute(0, 2, 1, 3).reshape(
            num_graph * self.num_head,
            src_node,
            self.head_dim)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # dim:[num_graph*num_head,tgt_node,src_node]

        attn_weights = attn_weights.reshape(num_graph, self.num_head, tgt_node, src_node)
        if attn_bias is not None:
            attn_weights += attn_bias
        #
        if attn_mask is not None:
            # print(attn_mask)
            attn_weights = attn_weights.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(1),
                float("-inf")
            )
        if padding_mask is not None:

            attn_weights = attn_weights.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        if self.dropout:
            attn_weights = self.Dropout(attn_weights).to(self.device)
        attn_weights = attn_weights.reshape(num_graph*self.num_head, tgt_node, src_node)
        attn_prob = torch.softmax(attn_weights, dim=-1)
        # print(attn_prob)
        attn = torch.bmm(attn_prob,v)
        # print(attn)
        attn = attn.reshape(num_graph,self.num_head,tgt_node,self.head_dim).permute(0,2,1,3).reshape(num_graph,tgt_node,self.hidden_dim)
        # dim:[num_graph,tgt_node,hidden_dim]

        return attn

class AttentionBias(nn.Module):
    def __init__(
            self,
            num_head,
            edge_type,
            max_dist,
            multi_hop_max_dist,
            hidden_dim,
            device="cuda:0"
    ):

        super(AttentionBias, self).__init__()
        self.device = torch.device(device)
        self.num_head = num_head
        self.edge_type = edge_type
        self.max_dist = max_dist
        self.multi_hop_max_dist = multi_hop_max_dist
        self.hidden_dim = hidden_dim
        if self.edge_type == 'multi_hop':
            self.adj_encoding = nn.Embedding(2,hidden_dim,padding_idx=0).to(self.device)
            self.bias_proj = nn.Embedding(1,max_dist*hidden_dim*num_head).to(self.device)
        else:
            self.adj_bias = nn.Embedding(2,num_head,padding_idx=0).to(self.device)

        self.virtual_token_bias = nn.Embedding(1,num_head).to(self.device)

    def forward(self,adj):
        num_graph,num_node = adj.shape[:2]
        graph_attn_bias = torch.zeros(num_graph,self.num_head,num_node+1,num_node+1,device=self.device)
        #dim: [num_graph, num_head, num_node+1, num_node+1]

        # virtual_token_bias
        virtual_token_bias = self.virtual_token_bias.weight.view(1,self.num_head,1)
        # dim: [1, num_head, 1]
        graph_attn_bias[:,:,0,:] += virtual_token_bias
        graph_attn_bias[:,:,1:,0] += virtual_token_bias

        # encoder bias
        bias = self.adj_bias(adj[:, :num_node, :num_node].long().to(self.device))
        bias = bias.permute(0,3,1,2)
        # if mask is not None:
        #     bias = bias * mask[:, None, None, None]

        graph_attn_bias[:, :, 1:, 1:] += bias

        return graph_attn_bias


class init_params():

    def __init__(self,beta=1):

        self.beta = beta
    def init_params(self,module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight,gain=self.beta)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # if isinstance(module, nn.Embedding):
        #     nn.init.constant_(module.weight,0)
        if isinstance(module, MultiHeadAttention):
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight,gain=self.beta)
            nn.init.xavier_uniform_(module.q_proj.weight)

        if isinstance(module,nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias,0)

class lambda_lr():

    def __init__(self,warmup_step,beta):
        self.warmup_step = warmup_step
        self.beta = beta
    def lambda_lr(self,last_step):
        if last_step < self.warmup_step:
            return (last_step+1)/self.warmup_step
        else:
            return math.exp(-(last_step+1-self.warmup_step)*self.beta)
