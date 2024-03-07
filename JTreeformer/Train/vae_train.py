import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn
import pickle
import math
import numpy as np
import os
import math
import argparse

from module.JTreeformer import JTreeformer
from Loss.Loss import VAE_Loss3
from module.jtreeformer_modules import lambda_lr
from torch.utils.data import DataLoader,Dataset

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# dataset
class JTDataSet(Dataset):
    def __init__(self,batch_data):
        self.adj = batch_data['adj'].long()
        self.x = batch_data['x'].long()
        self.hs = batch_data['hs'].long()
        self.layer_number = batch_data['layer_number'].long()
        self.property = batch_data['property']
    def __getitem__(self,index):
        return self.x[index],self.hs[index],self.layer_number[index],self.property[index],self.adj[index]

    def __len__(self):
        return self.x.shape[0]


class kl_coef():
    def __init__(self,size,batch_size,epoch):
        self.cnt=0
        self.total_step=math.ceil(size/batch_size)*epoch
        self.warm_up_step=int(self.total_step*0.5)
        self.raise_step=int(self.total_step*1.0)
        self.station_step=32

    def step(self):
        self.cnt+=1
    def coef(self):
        if self.cnt<self.warm_up_step:
            return 0
        elif self.cnt<self.raise_step:
            return 1/int((self.raise_step-self.warm_up_step)/self.station_step)*int((self.cnt-self.warm_up_step)/self.station_step)
            # return 1 /(self.raise_step - self.warm_up_step)*(self.cnt - self.warm_up_step)
        else:
            return 1

class VAE_Train():
    def __init__(self,device="cuda:0",epoch=5,mini_batch_size=256):
        self.device = torch.device(device)
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        self.loss1 = nn.CrossEntropyLoss(reduction="mean",ignore_index=0)
        self.loss2 = nn.CrossEntropyLoss(reduction="mean",ignore_index=0)

    def train(
            self,
            train_data:JTDataSet,
            valid_data:JTDataSet,
            model:JTreeformer,
            separate_train=True,
            alpha=1,
            gamma=0.2,
            cls_auxiliary=True,
            lr=0.001,
    ):
        model_path=os.path.join(os.path.abspath(''),"vae")
        if os.path.isdir(model_path) is False:
            os.makedirs(model_path)

        epoch_loss = np.array([])
        valid_loss=np.array([])
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
        size=train_data.__len__()

        lamb = lambda_lr(warmup_step=math.ceil(size//self.mini_batch_size*self.epoch*1/8),beta=math.log(500)/math.ceil(size//self.mini_batch_size*self.epoch*7/8))

        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lamb.lambda_lr)
            
        KL_coef=kl_coef(size=size,batch_size=self.mini_batch_size,epoch=self.epoch)
        loader=MultiEpochsDataLoader(train_data,self.mini_batch_size,shuffle=True,num_workers=4,drop_last=True)
        mini_batch_data=dict()

        for j in range(self.epoch):

            print("----------------epoch: "+str(j+1)+"----------------\n")
            for i,tensors in enumerate(loader):
                x,hs,layer_number,property,adj=tensors
                mini_batch_data['x']=x.to(self.device)
                mini_batch_data['hs']=hs.to(self.device)
                mini_batch_data['layer_number'] =layer_number.to(self.device)
                mini_batch_data['property'] = property.to(self.device)
                mini_batch_data['adj'] = adj.to(self.device)

                result_node,result_edge,mean_encoding,lnvar_encoding,z = model.forward(batch_data=mini_batch_data)
                pred_property=None
                if cls_auxiliary:
                    pred_property=model.encoder.property_proj(z)
                # gc.collect()
                # torch.cuda.empty_cache()
                Loss,mean_CE_node,mean_CE_edge,KL,w_loss,logp_loss,tpsa_loss = VAE_Loss3(
                    mini_batch_data,
                    result_node,
                    result_edge,
                    mean_encoding,
                    lnvar_encoding,
                    alpha,
                    pred_property=pred_property,
                    loss1=self.loss1,
                    loss2=self.loss2,
                    beta=0,
                    gamma=gamma,
                    cls_auxiliary=cls_auxiliary,
                    device=self.device)
                optimizer.zero_grad()
                # v_n = []
                # v_v = []
                # v_g = []
                # for name, parameter in model.named_parameters():
                #     v_n.append(name)
                #     v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                #     v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                # for k in range(len(v_n)):
                #     print('%s: %.3e ~ %.3e' % (v_n[k], np.min(v_v[k]).item(), np.max(v_v[k]).item()))
                #     print('%s: %.3e ~ %.3e' % (v_n[k], np.min(v_g[k]).item(), np.max(v_g[k]).item()))
                Loss.backward()
                optimizer.step()
                warmup.step()
                KL_coef.step()

                print(
                    f"data:{np.round((i+1)*self.mini_batch_size/size*100,3)}%,kl_coef:{KL_coef.coef()},loss:{np.round(Loss.item(), 5)},{np.round(mean_CE_node.item(), 5)},{np.round(mean_CE_edge.item(), 5)},{np.round(KL.item(), 5)},"
                    f"{np.round(w_loss.item(), 5)},{np.round(logp_loss.item(), 5)},{np.round(tpsa_loss.item(), 5)}")

                epoch_loss = np.append(epoch_loss,(Loss.item(),mean_CE_node.item(),mean_CE_edge.item(),KL.item(),w_loss.item(),logp_loss.item(),tpsa_loss.item()))
            if (j+1)%1==0:
                torch.save(model.state_dict(),os.path.join(model_path,f"vae_model_moses3_epoch{(j+1)}.pth"))

            with torch.no_grad():
                loader2=MultiEpochsDataLoader(valid_data,self.mini_batch_size*4,shuffle=True,num_workers=4,drop_last=True)
                mini_batch_data2=dict()
                for i, tensors in enumerate(loader2):
                    x, hs, layer_number, property, adj = tensors
                    mini_batch_data2['x'] = x.to(self.device)
                    mini_batch_data2['hs'] = hs.to(self.device)
                    mini_batch_data2['layer_number'] = layer_number.to(self.device)
                    mini_batch_data2['property'] = property.to(self.device)
                    mini_batch_data2['adj'] = adj.to(self.device)
                    result_node,result_edge,mean_encoding,lnvar_encoding,z = model.forward(batch_data=mini_batch_data2)
                    pred_property=None
                    if cls_auxiliary:
                        pred_property=model.encoder.property_proj(z)
                    try:
                        Loss,mean_CE_node,mean_CE_edge,KL,w_loss,logp_loss,tpsa_loss = VAE_Loss3(
                            mini_batch_data2,
                            result_node,
                            result_edge,
                            mean_encoding,
                            lnvar_encoding,
                            alpha,
                            pred_property=pred_property,
                            loss1=self.loss1,
                            loss2=self.loss2,
                            beta=0,
                            gamma=gamma,
                            cls_auxiliary=cls_auxiliary,
                            device=self.device)

                        print(f"epoch:{j} ,loss:{np.round(Loss.item(),5)},{np.round(mean_CE_node.item(),5)},{np.round(mean_CE_edge.item(),5)},{np.round(KL.item(),5)},"
                            f"{np.round(w_loss.item(),5)},{np.round(logp_loss.item(),5)},{np.round(tpsa_loss.item(),5)}")

                        valid_loss=np.append(valid_loss,(Loss.item(),mean_CE_node.item(),mean_CE_edge.item(),KL.item(),w_loss.item(),logp_loss.item(),tpsa_loss.item()))
                    except:
                        continue

        return epoch_loss,valid_loss


def pretrain_model(args):
    loss_path = os.path.join(os.path.abspath(''),"vae_loss")
    if os.path.isdir(loss_path) is False:
        os.makedirs(loss_path)
    model = JTreeformer(
            num_layers_encoder =args.num_layers_encoder,
            num_layers_decoder =args.num_layers_decoder,
            hidden_dim_encoder =args.hidden_dim_encoder,
            expand_dim_encoder =args.expand_dim_encoder,
            hidden_dim_decoder =args.hidden_dim_decoder,
            expand_dim_decoder=args.expand_dim_decoder,
            latent_space_dim=args.latent_space_dim,
            num_head_encoder=args.num_head_encoder,
            num_head_decoder=args.num_head_decoder,
            num_node_type=args.num_node_type,
            max_hs=args.max_hs,
            max_degree=args.max_degree,
            max_layer_num=args.max_layer_num,
            max_brother_num=args.max_brother_num,
            dropout=args.dropout,
            dropout_rate=args.dropout_rate,
            device=args.device,
            feature_test=args.feature_test,
            g_test=args.g_test
    )

    with open(args.train_path, "rb") as file:
        train_data = pickle.load(file)
    train_data=JTDataSet(train_data)
    with open(args.valid_path, "rb") as file:
        valid_data = pickle.load(file)
    valid_data=JTDataSet(batch_data=valid_data)
    print("data load done.")
    if args.cls_auxiliary:

        model.encoder.add_module('property_proj',
                                 nn.Linear(model.encoder.hidden_dim,4).to(
                                     args.device))
        nn.init.xavier_normal_(model.encoder.property_proj.weight, gain=0.87 * (((12+ 2) ** 4) *12) ** (-1 / 16))
        nn.init.constant_(model.encoder.property_proj.bias, 0)
    train=VAE_Train(mini_batch_size=args.batch_size,epoch=args.epoch,device = args.device)
    epoch_loss,valid_loss = train.train(
        train_data=train_data,valid_data=valid_data,
        model=model, lr=0.0001,separate_train=False,cls_auxiliary=args.cls_auxiliary)
    pickle.dump(epoch_loss,open(os.path.join(loss_path,"vae_train_loss_moses3.pkl"),"wb"))
    pickle.dump(valid_loss, open(os.path.join(loss_path,"vae_valid_loss_moses3.pkl"), "wb"))
    print("loss save done.")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_path',default=os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','MolDataSet',"train_data_moses.pkl"))
    parser.add_argument('--valid_path',default=os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','MolDataSet',"valid_data_moses.pkl"))
    parser.add_argument('--cls_auxiliary',default=True)
    parser.add_argument('--batch_size',default=256)
    parser.add_argument('--epoch',default=10)
    parser.add_argument('--num_layers_encoder',default= 12)
    parser.add_argument('--num_layers_decoder',default= 12)
    parser.add_argument('--hidden_dim_encoder',default= 512)
    parser.add_argument('--expand_dim_encoder',default= 1024)
    parser.add_argument('--hidden_dim_decoder',default= 768)
    parser.add_argument('--expand_dim_decoder',default= 1536)
    parser.add_argument('--latent_space_dim',default= 768)
    parser.add_argument('--num_head_encoder',default= 16)
    parser.add_argument('--num_head_decoder',default= 16)
    parser.add_argument('--num_node_type',default= 710)
    parser.add_argument('--max_hs',default= 50)
    parser.add_argument('--max_degree',default= 20)
    parser.add_argument('--max_layer_num',default= 50)
    parser.add_argument('--max_brother_num',default= 20)
    parser.add_argument('--dropout',default= True)
    parser.add_argument('--dropout_rate',default= 0.1)
    parser.add_argument('--device', default = "cuda:5")
    parser.add_argument('--feature_test', default =False)
    parser.add_argument('--g_test', default =False)
    args=parser.parse_args()
    pretrain_model(args)
    # finetune_model()
