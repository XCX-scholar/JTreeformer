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
            device="cuda:0",
    ):
        super(MultiHeadAttention, self).__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.k_proj = nn.Linear(self.k_dim, hidden_dim, bias=require_bias).to(self.device)
        self.q_proj = nn.Linear(self.q_dim, hidden_dim, bias=require_bias).to(self.device)
        self.v_proj = nn.Linear(self.v_dim, hidden_dim, bias=require_bias).to(self.device)
        self.register_buffer('kv_cache_key', None)
        self.register_buffer('kv_cache_value', None)

        # Dropout
        self.dropout = dropout
        if self.dropout:
            self.Dropout = nn.Dropout(p=dropout_rate)

    def _reset_cache(self):
        self.kv_cache_key = None
        self.kv_cache_value = None

    def forward(self,x,attn_bias=None,attn_mask=None,padding_mask=None,key=None,value=None,use_kv_cache=False):
        num_graph, tgt_node, hidden_dim = x.size()
        if use_kv_cache:
            if self.self_attention:
                if self.kv_cache_key is not None:
                    expected_seq_len = self.kv_cache_key.size(1) + tgt_node
                    if attn_mask is not None and attn_mask.size(-1) != expected_seq_len:
                        raise ValueError("Attention mask length does not match cached sequence length")
                    key = torch.cat([self.kv_cache_key, self.k_proj(x)], dim=1)
                    value = torch.cat([self.kv_cache_value, self.v_proj(x)], dim=1)
                else:
                    key = self.k_proj(x)
                    value = self.v_proj(x)
                self.kv_cache_key = key.detach()
                self.kv_cache_value = value.detach()
            else:
                raise ValueError("Only support self-attention KV-cache.")
        else:
            if not self.self_attention:
                if key is not None and value is not None:
                    key = self.k_proj(key)
                    value = self.v_proj(value)
                else:
                    raise ValueError("Key,value should not be provided for cross-attention")
            else:
                key = self.k_proj(x)
                value = self.v_proj(x)
        q = self.q_proj(x)
        q *= self.scaling
        q = q.view(num_graph, tgt_node, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(num_graph, -1, self.num_head, self.head_dim).permute(0, 2, 3, 1)  # [B, H, D, S]
        value = value.view(num_graph, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, key)  # [B, H, T, S]
        if attn_bias is not None:
            attn_weights += attn_bias

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(1),
                float("-inf")
            )
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        attn_weights = torch.softmax(attn_weights, dim=-1)

        if self.dropout:
            attn_weights = self.Dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(num_graph, tgt_node, self.hidden_dim)

        return attn_output

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
