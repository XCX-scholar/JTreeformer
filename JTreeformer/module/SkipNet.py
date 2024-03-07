import torch
from torch import nn
import torch.nn.functional as F
import math


class TimeEmbeddings(nn.Module):
    def __init__(self, dim, device="cuda:0"):
        super().__init__()
        self.dim = dim
        self.device = torch.device(device)

    def forward(self,times):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings).to(self.device)
        embeddings = times[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class Block(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        time_embedding_dim,
        dropout=True,
        dropout_rate=0.2,
        device="cuda:0",
    ):

        super(Block,self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.time_embedding_dim = time_embedding_dim
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.device = torch.device(device)
        self.Dropout = nn.Dropout(p=dropout_rate)


        self.embedding_layers = nn.Sequential(
            nn.Linear(
                self.time_embedding_dim,
                self.out_dim,
            ).to(self.device),
            nn.SiLU(),
            nn.Linear(
                self.out_dim,
                self.out_dim,
            ).to(self.device),
        )
        self.ff_ln = nn.LayerNorm(out_dim).to(self.device)
        self.ff = nn.Linear(in_dim,out_dim).to(self.device)

    def forward(self, x, time_embedding):
        assert x.shape[-1] == self.in_dim
        assert time_embedding.shape[-1] == self.time_embedding_dim
        embedding_out = self.embedding_layers(time_embedding)
        x=self.ff(x)
        x = x*embedding_out
        x = F.silu(x)
        x = self.ff_ln(x)
        if self.Dropout:
            x=self.Dropout(x)

        return x

class SkipNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        time_embedding_dim,
        num_block=6,
        dropout=True,
        dropout_rate=0.2,
        Init_params=True,
        device="cuda:0"
    ):

        super(SkipNet,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_block = num_block
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.device = torch.device(device)
        self.Init_params = Init_params

        self.time_embedding = TimeEmbeddings(dim=self.time_embedding_dim,device=device)
        self.in_layer = Block(
            in_dim=in_dim,
            out_dim=hidden_dim,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
            dropout_rate=dropout_rate,
            device=device
        )
        self.middle_blocks = nn.ModuleList([
            Block(
                in_dim=hidden_dim+in_dim,
                out_dim=hidden_dim,
                time_embedding_dim=time_embedding_dim,
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

    def forward(self, x, timesteps):
        time_embedding = self.time_embedding(timesteps)
        r=x
        x=self.in_layer(x,time_embedding)
        for layer in self.middle_blocks:
            x=torch.cat([x,r],dim=-1)
            x=layer(x,time_embedding)
        x=self.out_layer(torch.cat([x,r],dim=-1))

        return x+r