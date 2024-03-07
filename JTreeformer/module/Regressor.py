import torch
from torch import nn
import torch.nn.functional as F
import math
from module.SkipNet import SkipNet



class Regressor(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        time_embedding_dim,
        num_block=3,
        dropout=True,
        dropout_rate=0.2,
        Init_params=True,
        device="cuda:0"
    ):
        super(Regressor, self).__init__()
        self.Net=SkipNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            time_embedding_dim=time_embedding_dim,
            num_block=num_block,
            dropout=dropout,
            dropout_rate=dropout_rate,
            Init_params=Init_params,
            device=device
        )
        self.reg=nn.Linear(out_dim,4).to(device)

    def forward(self,x,t):
        x=self.Net(x,t)
        x=self.reg(F.relu(x))
        return x