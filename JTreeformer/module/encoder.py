from module.jtreeformer_modules import MultiHeadAttention,init_params
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self,
                 alpha,
                 hidden_dim = 512,
                 num_head = 8,
                 sandwich_ln = False,
                 expand_dim = 1024,
                 device="cuda:0"
                 ):
        super(EncoderLayer,self).__init__()


        self.alpha=alpha
        self.device=torch.device(device)
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.sandwich_ln = sandwich_ln
        self.expand_dim = expand_dim

        self.attn_ln = F.layer_norm
        self.ff_ln = F.layer_norm
        if self.sandwich_ln == True:
            self.attn_ln_sandwich1 = F.layer_norm
            self.attn_ln_sandwich2 = F.layer_norm
            self.ff_ln_sandwich1 = F.layer_norm
            self.ff_ln_sandwich2 = F.layer_norm

        self.self_attn = MultiHeadAttention(hidden_dim=hidden_dim,
                                            num_head=num_head,
                                            k_dim=hidden_dim,
                                            v_dim=hidden_dim,
                                            q_dim=hidden_dim,
                                            self_attention=True,
                                            device=device,)
        self.gcn_proj=nn.Linear(hidden_dim,hidden_dim).to(self.device)
        self.fusion = nn.Linear(2*hidden_dim,hidden_dim).to(self.device)
        self.ff1 = nn.Linear(hidden_dim,expand_dim).to(self.device)
        self.ff2 = nn.Linear(expand_dim,hidden_dim).to(self.device)

    def forward(self,x,T,attn_mask,attn_bias,padding_mask):

        residual = x
        if self.sandwich_ln:
            x = self.attn_ln_sandwich1(x,[self.hidden_dim,])
        x1 = self.self_attn(
            x=x,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            padding_mask=padding_mask
        )
        x2 = F.relu(self.gcn_proj(torch.bmm(T,x)))
        x=torch.cat([x1,x2],dim=-1)
        x = self.fusion(x)
        if self.sandwich_ln:
            x = self.attn_ln_sandwich2(x,[self.hidden_dim,])
        x = self.alpha*residual + x
        if not self.sandwich_ln:
            x = self.attn_ln(x,[self.hidden_dim,])

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

class Encoder(nn.Module):
    def __init__(
        self,
        num_node_type,
        alpha,
        beta,
        num_layers = 12,
        hidden_dim = 512,
        expand_dim = 1024,
        num_head = 16,
        encoder_normalize_before = True,
        apply_init = True,
        embed_scale: float = None,
        sandwich_ln = False,
        dropout=True,
        in_training=True,
        dropout_rate=0.1,
        device="cuda:0"
    ):

        super(Encoder,self).__init__()
        self.device = torch.device(device)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embed_scale = embed_scale
        self.apply_init = apply_init
        self.in_training = in_training
        if encoder_normalize_before:
            self.embed_ln = F.layer_norm
        else:
            self.embed_ln = None

        self.layers = nn.ModuleList([EncoderLayer(
                                    alpha=alpha,
                                    hidden_dim=hidden_dim,
                                    num_head=num_head,
                                    expand_dim=expand_dim,
                                    sandwich_ln=sandwich_ln,
                                    device=device)
                                     for _ in range(num_layers)])
        self.node_proj = nn.Linear(hidden_dim,num_node_type+2).to(self.device)

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
            attn_bias,
            padding_mask,
            perturb=None,
            attn_mask = None,
    ):
        if perturb is not None:
            x[:,1:,:] += perturb
        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.embed_ln is not None:
            x = self.embed_ln(x,[self.hidden_dim,])

        cnt = 0
        for layer in self.layers:
            if self.dropout:
                x = self.Dropout[cnt](x)
                cnt += 1
            x = layer(x,T,padding_mask=padding_mask,attn_mask=attn_mask,attn_bias=attn_bias)
        if self.in_training:
            neighbor_node_logit = self.node_proj(x[:,1:,:])

            return neighbor_node_logit,x[:,0,:]
        else:
            return x[:,0,:]
