from module.jtreeformer_modules import NodeFeature,AttentionBias,init_params,MultiHeadAttention
from module.encoder import Encoder
from module.decoder import Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class JTreeformer(nn.Module):
    def __init__(
            self,
            num_layers_encoder = 12,
            num_layers_decoder = 12,
            hidden_dim_encoder = 512,
            expand_dim_encoder = 1024,
            hidden_dim_decoder = 768,
            expand_dim_decoder=1536,
            latent_space_dim=768,
            num_head_encoder=16,
            num_head_decoder=16,
            num_node_type=710,
            max_hs=50,
            max_degree=20,
            max_layer_num=200,
            max_brother_num=20,
            dropout=True,
            dropout_rate=0.1,
            device="cuda",
            feature_test=False,
            g_test=False
    ):

        super(JTreeformer, self).__init__()
        self.device = torch.device(device)
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.latent_space_dim=latent_space_dim

        self.encoder_node_feature = NodeFeature(
            num_node_type=num_node_type,
            max_hs=max_hs,
            max_degree=max_degree,
            max_layer_num=max_layer_num,
            max_brother_num=max_brother_num,
            hidden_dim=hidden_dim_encoder,
            device=device,
            feature_test=feature_test
        )

        self.decoder_node_feature = NodeFeature(
            num_node_type=num_node_type,
            max_hs=max_hs,
            max_degree=max_degree,
            max_layer_num=max_layer_num,
            max_brother_num=max_brother_num,
            hidden_dim=hidden_dim_decoder,
            device=device,
            decoder=True,
            feature_test=feature_test
        )


        self.encoder_bias = AttentionBias(
            num_head=num_head_encoder,
            edge_type='adj',
            max_dist=64,
            multi_hop_max_dist=0,
            hidden_dim=hidden_dim_encoder,
            device=device
        )

        self.decoder_bias = AttentionBias(
            num_head=num_head_decoder,
            edge_type='adj',
            max_dist=64,
            multi_hop_max_dist=0,
            hidden_dim=hidden_dim_decoder,
            device=device
        )

        self.encoder = Encoder(num_node_type=num_node_type,
                               alpha=0.81*(((num_layers_encoder+2)**4)*num_layers_decoder)**(1/16),
                               beta=0.87*(((num_layers_encoder+2)**4)*num_layers_decoder)**(-1/16),
                               num_layers=num_layers_encoder,
                               hidden_dim=hidden_dim_encoder,
                               expand_dim=expand_dim_encoder,
                               num_head=num_head_encoder,
                               device=device,
                               dropout=dropout,
                               dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers_decoder,
                               alpha=(2*num_layers_decoder+1)**(1/4),
                               beta=(8*num_layers_decoder+4)**(-1/4),
                                num_node_type=num_node_type,
                                hidden_dim=hidden_dim_decoder,
                                expand_dim=expand_dim_decoder,
                                device=device,
                                num_head=num_head_decoder,
                                dropout=dropout,
                                g_test=g_test)

        self.adj_feature = nn.Embedding(num_node_type + 2, hidden_dim_decoder, padding_idx=0).to(self.device)

        self.adj_cross_attn=MultiHeadAttention(
            self_attention=False,
            hidden_dim=hidden_dim_decoder,
            num_head=num_head_decoder,
            k_dim=hidden_dim_decoder,
            v_dim=hidden_dim_decoder,
            q_dim=hidden_dim_decoder,
            device=device
        )

        self.relation_logit_proj=nn.Linear(hidden_dim_decoder,4).to(self.device)

        self.mean_proj = nn.Linear(hidden_dim_encoder,latent_space_dim).to(self.device)
        self.lnvar_proj = nn.Linear(hidden_dim_encoder,latent_space_dim).to(self.device)

        self.fusion_attn=MultiHeadAttention(
            hidden_dim=hidden_dim_decoder,
            num_head=num_head_decoder,
            k_dim=hidden_dim_decoder+latent_space_dim,
            v_dim=hidden_dim_decoder+latent_space_dim,
            q_dim=hidden_dim_decoder+latent_space_dim,
            device=device
        )

        self.thetas=nn.Parameter(torch.ones(num_layers_decoder,device=self.device,requires_grad=True))
        init = init_params(beta=0.87*(((num_layers_encoder+2)**4)*num_layers_decoder)**(-1/16))
        init.init_params(self.mean_proj)
        init.init_params(self.lnvar_proj)
        init = init_params(beta=(8 * num_layers_decoder + 4) ** (-1 / 4))
        init.init_params(self.fusion_attn)
        init.init_params(self.adj_cross_attn)
        init.init_params(self.relation_logit_proj)

    def forward(self,batch_data,encoding_only=False):

        x=batch_data['x'].to(self.device)
        adj=batch_data['adj'].to(self.device)
        layer_number=batch_data['layer_number'].to(self.device)
        hs=batch_data['hs'].to(self.device)
        num_graph,num_node = x.shape[:2]


        attn_mask = torch.triu(torch.ones(num_node+1,num_node+1),diagonal=1).bool().to(self.device)
        padding_mask = (x[:, :]).eq(0).to(self.device)
        # dim[num_graph,num_node,1]
        padding_mask_cls = torch.zeros(num_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1).to(self.device)


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
        bro_ord = torch.gather(bro_ord, dim=-1, index=(father_position - 1).clamp(min=0).unsqueeze(-1)).reshape(num_graph,num_node)
        bro_ord[flag.eq(1)] = 0
        bro_ord=bro_ord.long()

        adj_with_cls = torch.cat([torch.ones((num_graph, 1,num_node),device=self.device, dtype=adj.dtype),adj],dim=1)
        adj_with_cls = torch.cat([torch.ones((num_graph,num_node+1,1),device=self.device, dtype=adj.dtype),adj_with_cls], dim=2)
        adj_with_cls[:,0,0]=0
        adj_with_cls=adj_with_cls+torch.eye(num_node+1,num_node+1,device=self.device).unsqueeze(0)

        degree=adj.sum(dim=-1)
        degree_with_cls=torch.cat([degree.clamp(0,1).sum(dim=-1).unsqueeze(-1),degree], dim=1)
        degree_diag=torch.diag_embed(torch.sqrt(1/(degree_with_cls+1)))
        T=torch.bmm(adj_with_cls,degree_diag)
        T=torch.bmm(degree_diag,T)


        encoder_bias = self.encoder_bias(adj)
        # print("encoder bias finished")
        encoder_node_feature = self.encoder_node_feature(
            x=x,
            hs=hs,
            degree=degree.long(),
            father_position=father_position,
            brother_order=bro_ord,
            layer_number=layer_number.long()
        )
        # print("encoder node feature finished")


        self.encoder.in_training=False
        z=F.relu(self.encoder(x=encoder_node_feature,T=T,padding_mask=padding_mask,attn_bias=encoder_bias))
        # print("encoding finished")

        mean_encoding = self.mean_proj(z)
        lnvar_encoding = self.lnvar_proj(z)
        if encoding_only:
            return mean_encoding,lnvar_encoding,z
        else:
            encoding_result = mean_encoding + torch.randn_like(lnvar_encoding) * torch.exp(lnvar_encoding * 0.5)

            adj_with_cls = torch.cat([torch.ones(num_graph,num_node,1,device=self.device, dtype=adj.dtype),masked_adj], dim=2)
            adj_with_cls = torch.cat([torch.zeros(num_graph,1,num_node+1,device=self.device, dtype=adj.dtype),adj_with_cls],dim=1)

            degree=masked_adj.sum(dim=-1)
            degree_with_cls=torch.cat([torch.zeros(num_graph,1,device=self.device),degree], dim=1)
            degree_diag=torch.diag_embed(torch.sqrt(1/(degree_with_cls+1)))

            T=torch.diag_embed(degree_with_cls+1)-adj_with_cls
            T=torch.bmm(degree_diag,T)
            T=torch.bmm(T,degree_diag)

            decoder_bias = self.decoder_bias(adj)
            # print("decoder bias finished")


            decoder_node_feature = self.decoder_node_feature(
                x=x,
                hs=hs,
                father_position=father_position,
                brother_order=bro_ord,
                layer_number=layer_number.long()
            )

            decoder_node_feature = torch.cat([decoder_node_feature,encoding_result.unsqueeze(1).repeat(1, num_node + 1, 1)],
                                        dim=-1)
            decoder_node_feature=self.fusion_attn(x=decoder_node_feature,attn_mask=attn_mask,padding_mask=padding_mask)

            result_node,node_decoding = self.decoder(x=decoder_node_feature,T=T,thetas=self.thetas,attn_mask=attn_mask,padding_mask=padding_mask,attn_bias=decoder_bias)

            decoder_adj_feature=self.adj_feature(x)
            decoder_adj_feature=self.adj_cross_attn(
                x=node_decoding[:,:num_node],key=decoder_adj_feature,
                value=decoder_adj_feature,
                attn_mask=attn_mask[:num_node,:num_node],
                padding_mask=padding_mask[:,1:]
            )
            relation_logit=self.relation_logit_proj(decoder_adj_feature)

            # print("decoding finished")

            return result_node,relation_logit,mean_encoding,lnvar_encoding,z
