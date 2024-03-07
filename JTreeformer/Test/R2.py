import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.JTreeformer import JTreeformer
import torch
import pickle
import torch.nn as nn
from Train.vae_train import JTDataSet,MultiEpochsDataLoader

valid_filename = os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','MolDataSet','valid_data_moses.pkl')
model_filename = os.path.join(os.path.abspath(''),'vae',"vae_model_moses_tt_epoch4.pth")
model = JTreeformer(
    num_layers_encoder=12,
    num_layers_decoder=12,
    hidden_dim_encoder=512,
    expand_dim_encoder=1024,
    hidden_dim_decoder=768,
    expand_dim_decoder=1536,
    latent_space_dim=768,
    num_head_encoder=16,
    num_head_decoder=16,
    num_node_type=710,
    max_hs=50,
    max_degree=20,
    max_layer_num=50,
    max_brother_num=20,
    dropout=True,
    dropout_rate=0.1,
    device='cpu',
    feature_test=True,
    g_test=True
)
model.encoder.add_module('property_proj', nn.Linear(model.encoder.hidden_dim, 4).to(
    torch.device('cpu')))
model.load_state_dict(torch.load(model_filename,map_location=torch.device('cpu')))

model.decoder.dropout=False

with open(valid_filename, "rb") as file:
    valid_data = pickle.load(file)
properties=valid_data['property']
mean=properties.mean(dim=0)
SST=torch.square(properties-mean.unsqueeze(0)).sum(dim=0)[:3]
valid_data=JTDataSet(batch_data=valid_data)
SSR=torch.zeros(3)
with torch.no_grad():
    loader2=MultiEpochsDataLoader(valid_data,1024,shuffle=True,num_workers=4,drop_last=True)
    mini_batch_data2=dict()
    for i, tensors in enumerate(loader2):
        print(i)
        x, hs, layer_number, property, adj = tensors
        mini_batch_data2['x'] = x
        mini_batch_data2['hs'] = hs
        mini_batch_data2['layer_number'] = layer_number
        mini_batch_data2['adj'] = adj
        result_node,result_edge,mean_encoding,lnvar_encoding,z = model.forward(batch_data=mini_batch_data2)
        pred_property=None
        pred_property=model.encoder.property_proj(z)
        SSR+=torch.square(pred_property[:,:3]-property[:,:3]).sum(dim=0)

print(SSR/SST)