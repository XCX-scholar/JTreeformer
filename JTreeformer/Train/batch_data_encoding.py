import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import pickle
from Train.vae_train import JTDataSet
from torch.utils.data import DataLoader
from module.JTreeformer import JTreeformer
import torch.nn as nn

train_filename = os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','MolDataSet','train_data_moses.pkl')
valid_filename = os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','MolDataSet','valid_data_moses.pkl')
model_filename = os.path.join(os.path.abspath(''),'vae',"vae_model_moses3_epoch6.pth")
train_save_name=os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','MolDataSet','train_encoding_moses3.pkl')
valid_save_name=os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','MolDataSet','1div3_valid_encoding_moses3.pkl')
# folder="D:\\MolDataSet\\new_batch_data\\"
# model_path=""

# with open(train_filename, "rb") as file:
#     train_data = pickle.load(file)
with open(valid_filename, "rb") as file:
    train_data = pickle.load(file)
train_data = JTDataSet(train_data)
print("load done.")
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
    device="cpu",
    feature_test=False,
    g_test=False
)
model.encoder.add_module('property_proj', nn.Linear(model.encoder.hidden_dim, 4).to(
    torch.device("cpu")))
model.load_state_dict(torch.load(model_filename,map_location=torch.device('cpu')))
print("load done.")

train_e = torch.zeros(0, 768, device=torch.device("cpu"))
train_property = torch.zeros(0, 3, device=torch.device("cpu"))

# torch.cuda.empty_cache()

mini_batch_data = dict()
model.encoder.dropout = False
with torch.no_grad():
    train_loader = DataLoader(train_data, batch_size=2048)
    for i, tensors in enumerate(train_loader):
        x, hs, layer_number, property, adj = tensors
        print(i)
        mini_batch_data['x'] = x.to("cpu")
        mini_batch_data['hs'] = hs.to("cpu")
        mini_batch_data['layer_number'] = layer_number.to("cpu")
        mini_batch_data['property'] = property.to("cpu")
        mini_batch_data['adj'] = adj.to("cpu")
        mean_encoding,lnvar_encoding,_ = model(mini_batch_data, encoding_only=True)
        tmp=mean_encoding + torch.randn_like(lnvar_encoding) * torch.exp(lnvar_encoding * 0.5)
        train_e = torch.cat([train_e, tmp], dim=0)
        train_property = torch.cat([train_property.to("cpu"), property.to("cpu")], dim=0)
    batch_data = dict()
    batch_data['x'] = train_e
    batch_data['property'] = train_property
    # pickle.dump(batch_data,open(train_save_name, "wb"))
    pickle.dump(batch_data,open(valid_save_name, "wb"))

    print(train_e.shape)