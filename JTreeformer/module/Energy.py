import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout=True,
        dropout_rate=0.2,
        device="cuda:0",
    ):

        super(Block,self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.device = torch.device(device)
        self.Dropout = nn.Dropout(p=dropout_rate)

        self.ff_ln = nn.LayerNorm(out_dim).to(self.device)
        self.ff = nn.Linear(in_dim,out_dim).to(self.device)

    def forward(self, x):
        assert x.shape[-1] == self.in_dim
        x=self.ff(x)
        x = F.silu(x)
        x = self.ff_ln(x)
        if self.Dropout:
            x=self.Dropout(x)

        return x

class Regressor(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        num_block=3,
        dropout=True,
        dropout_rate=0.2,
        Init_params=True,
        device="cuda:0"
    ):

        super(Regressor,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_block = num_block
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.device = torch.device(device)
        self.Init_params = Init_params

        self.in_layer = Block(
            in_dim=in_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            dropout_rate=dropout_rate,
            device=device
        )
        self.middle_blocks = nn.ModuleList([
            Block(
                in_dim=hidden_dim+in_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                dropout_rate=dropout_rate,
                device=device
            ) for i in range(num_block-2)
        ])

        self.out_layer = nn.Linear(hidden_dim+in_dim,out_dim).to(self.device)

        if self.Init_params:
            self.apply(self.init_params)

    def init_params(self,module):
        if isinstance(module,nn.Linear):
            nn.init.kaiming_normal_(module.weight,a=0,nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        r=x
        x=self.in_layer(x)
        for layer in self.middle_blocks:
            x=torch.cat([x,r],dim=-1)
            x=layer(x)
        x=self.out_layer(torch.cat([x,r],dim=-1))

        return x

class Energy(nn.Module):
    def __init__(self,
                 num_prop,
                 hidden_dim,
                 in_dim,
                 num_block,
                 device="cuda:0"):
        super(Energy, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_feature=num_prop
        self.device=device
        self.regressor=Regressor(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_block=num_block,
            dropout=True,
            dropout_rate=0.2,
            Init_params=True,
            device=device
        )
        self.W12=nn.Parameter(torch.randn(hidden_dim,num_prop,device=device,requires_grad=True))
        self.w1=nn.Parameter(torch.randn(hidden_dim,device=device,requires_grad=True))
        self.w2=nn.Parameter(torch.randn(num_prop,device=device,requires_grad=True))
        self.w3 = nn.Parameter(torch.randn(hidden_dim, device=device, requires_grad=True))
        self.w4 = nn.Parameter(torch.randn(num_prop, device=device, requires_grad=True))
        self.w=nn.Parameter(torch.randn(num_prop,device=device,requires_grad=True))
        self.b=nn.Parameter(torch.randn(num_prop, device=device, requires_grad=True))

    def forward(self,x,tgt_prop):
        hidden_unit=self.regressor(x)
        hidden_unit=torch.sigmoid(hidden_unit)
        tgt_prop=torch.sigmoid(tgt_prop*self.w.unsqueeze(0)+self.b.unsqueeze(dim=0))
        e=torch.einsum('ij,bj->bi',self.W12,tgt_prop)
        e=(hidden_unit*e).sum(dim=-1)+(self.w1.unsqueeze(0)*hidden_unit).sum(-1)+(self.w2.unsqueeze(0)*tgt_prop).sum(-1)+(self.w3.unsqueeze(0)*torch.square(hidden_unit)).sum(-1)+(self.w4.unsqueeze(0)*torch.square(tgt_prop)).sum(-1)
        return e

    def hard_neg_energy(self,x,hard_neg_prop):
        hidden_unit=self.regressor(x)
        hidden_unit=torch.sigmoid(hidden_unit)
        hard_neg_prop=torch.sigmoid(hard_neg_prop*self.w.unsqueeze(0).unsqueeze(-1)+self.b.unsqueeze(dim=0).unsqueeze(-1))
        es=torch.einsum('ij,bjk->bik',self.W12,hard_neg_prop)
        es=(hidden_unit.unsqueeze(-1)*es).sum(dim=-2)+(self.w1.unsqueeze(0)*hidden_unit).sum(-1).unsqueeze(-1)+(self.w2.unsqueeze(0).unsqueeze(-1)*hard_neg_prop).sum(-2)
        +(self.w3.unsqueeze(0) * torch.square(hidden_unit)).sum(-1).unsqueeze(-1) + (
                    self.w4.unsqueeze(0).unsqueeze(-1) * torch.square(hard_neg_prop)).sum(-2)
        hne,_=es.min(dim=-1)
        return hne

    def energy(self,x,prop_list):
        hidden_unit=self.regressor(x)
        tgt_prop=[]
        idx=[]
        n_idx=[]
        for i,prop in enumerate(prop_list):
            if prop is not None:
                tgt_prop.append(prop)
                idx.append(i)
            else:
                n_idx.append(i)
        tgt_prop=torch.Tensor(tgt_prop,device=self.device).unsqueeze(0)
        hidden_unit=torch.sigmoid(hidden_unit)
        tgt_prop=torch.sigmoid(tgt_prop*self.w[idx].unsqueeze(dim=0)+self.b[idx].unsqueeze(dim=0))
        e=torch.einsum('ij,bj->bi',self.W12[:,idx],tgt_prop)
        e=(hidden_unit*e).sum(dim=-1)+(self.w1.unsqueeze(0)*hidden_unit).sum(-1)+(self.w2[idx].unsqueeze(0)*tgt_prop).sum(-1)
        r=torch.einsum('ij,j->i',torch.square(self.W12[:,n_idx]),1/self.w4[n_idx]/4)
        e=e+((self.w3-r).unsqueeze(0)*torch.square(hidden_unit)).sum(-1)+(self.w4[idx]*torch.square(tgt_prop)).unsqueeze(0).sum(-1)
        return e
