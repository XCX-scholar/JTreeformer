from module.jtreeformer_modules import MultiHeadAttention,init_params
import torch
import torch.nn as nn
import torch.nn.functional as F
"""1622016 129024"""
'''720896 57344'''

class DecoderLayer(nn.Module):
    def __init__(self,
                 alpha,
                 hidden_dim=768,
                 num_head=16,
                 sandwich_ln=False,
                 expand_dim=1536,
                 device="cuda:0",
                 g_test=False):
        super(DecoderLayer, self).__init__()

        self.alpha=alpha
        self.device = torch.device(device)
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.sandwich_ln = sandwich_ln
        self.expand_dim = expand_dim

        self.self_attn_ln = F.layer_norm
        self.ff_ln = F.layer_norm
        if self.sandwich_ln == True:
            self.self_attn_ln_sandwich1 = F.layer_norm
            self.self_attn_ln_sandwich2 = F.layer_norm
            self.ff_ln_sandwich1 = F.layer_norm
            self.ff_ln_sandwich2 = F.layer_norm

        self.self_attn = MultiHeadAttention(hidden_dim=hidden_dim,
                                            num_head=num_head,
                                            k_dim=hidden_dim,
                                            v_dim=hidden_dim,
                                            q_dim=hidden_dim,
                                            self_attention=True,
                                            device=device)
        self.g_test=g_test
        if not self.g_test:
            self.dagcn_proj=nn.Linear(hidden_dim,hidden_dim).to(self.device) #DAGCN模块(我们论文中提出)的线性层
            self.fusion = nn.Linear(2*hidden_dim,hidden_dim).to(self.device) #两个模块拼接后映射回原本的维度，再残差链接
        self.ff1 = nn.Linear(hidden_dim, expand_dim).to(self.device)
        self.ff2 = nn.Linear(expand_dim, hidden_dim).to(self.device)

    def forward(self,x,T,attn_bias,attn_mask,padding_mask):

        residual = x
        if self.sandwich_ln:
            x = self.self_attn_ln_sandwich1(x,[self.hidden_dim,])
        x1 = self.self_attn(
            x=x,
            attn_bias=attn_bias,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )

        if not self.g_test:
            x2 = F.relu(self.dagcn_proj(torch.bmm(T,x)))
            x=torch.cat([x1,x2],dim=-1)
            x = self.fusion(x)
        else:
            x=x1
        if self.sandwich_ln:
            x = self.self_attn_ln_sandwich2(x,[self.hidden_dim,])
        x = self.alpha*residual + x
        if not self.sandwich_ln:
            x = self.self_attn_ln(x,[self.hidden_dim,])

        residual = x
        if self.sandwich_ln:
            x = self.ff_ln_sandwich1(x,[self.hidden_dim,])
        x = F.relu(self.ff1(x))
        x = self.ff2(x)
        if self.sandwich_ln:
            x = self.ff_ln_sandwich2(x,[self.hidden_dim,])
        x = self.alpha*residual + x
        if not self.sandwich_ln:
            x = self.ff_ln(x,[self.hidden_dim,])
        return x

class Decoder(nn.Module):
    def __init__(
            self,
            num_node_type,
            alpha,
            beta,
            num_layers = 12,
            hidden_dim=768,
            expand_dim =1536,
            num_head=16,
            latent_space_dim=512,
            decoder_normalize_before = True,
            apply_init = True,
            learned_pos_embedding = True,
            embed_scale: float = None,
            sandwich_ln = False,
            dropout = True,
            dropout_rate = 0.1,
            device="cuda:0",
            g_test=False
    ):

        super(Decoder, self).__init__()
        self.device = torch.device(device)
        self.num_layers = num_layers
        self.num_node_type = num_node_type
        self.hidden_dim = hidden_dim
        self.latent_space_dim=latent_space_dim
        self.embed_scale = embed_scale
        self.apply_init = apply_init
        self.learned_pos_embedding = learned_pos_embedding
        self.g_test=g_test
        if decoder_normalize_before:
            self.embed_ln = F.layer_norm
        else:
            self.embed_ln = None
        self.layers = nn.ModuleList([DecoderLayer(alpha=alpha,
                                                  hidden_dim=hidden_dim,
                                                  num_head=num_head,
                                                  expand_dim=expand_dim,
                                                  sandwich_ln=sandwich_ln,
                                                  device=device,
                                                  g_test=g_test)
                                     for _ in range(num_layers)])

        self.node_logit_proj = nn.Linear(hidden_dim,num_node_type+2).to(self.device) # 映射为节点类型的logit
        self.dropout = dropout
        if self.dropout:
            self.Dropout = [nn.Dropout(p=i.item()) for i in torch.arange(start=0,end=(1+1/self.num_layers)*dropout_rate,step=1/self.num_layers*dropout_rate)]


        if self.apply_init:
            init = init_params(beta=beta)
            self.apply(init.init_params)

    def forward(
            self,
            x,
            T,
            thetas,
            attn_mask,
            attn_bias,
            padding_mask,
            perturb=None,
    ):
        if perturb is not None:
            x[:, 1:, :] += perturb
        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.embed_ln is not None:
            x = self.embed_ln(x,[self.hidden_dim,])
        # print(x.shape)
        num_graph,num_node=x.shape[:2]

        cnt = 0
        for i,layer in enumerate(self.layers):
            if self.dropout:
                x = self.Dropout[cnt](x)
                cnt+=1
            x = layer(
                x=x,
                T=thetas[i]*T+torch.eye(num_node,num_node,device=self.device).unsqueeze(0),
                padding_mask=padding_mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
        node_logit=F.relu(x)
        node_logit = self.node_logit_proj(node_logit)

        return node_logit,x