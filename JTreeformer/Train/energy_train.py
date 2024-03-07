import torch
import torch.nn as nn
import pickle
import math
import numpy as np
import os
import math

from module.Energy import Energy
from module.DDPM import *
from Loss.Loss import E_Loss
from module.jtreeformer_modules import lambda_lr
from torch.utils.data import DataLoader,Dataset

folder = "MolFile/"
# folder="D:\\MolDataSet\\"

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
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# dataset
class EDataSet(Dataset):
    def __init__(self,batch_data):
        self.x = batch_data['x'].long()
        self.property = batch_data['property']
    def __getitem__(self,index):
        return self.x[index],self.property[index]

    def __len__(self):
        return self.x.shape[0]

class Energy_Train():
    def __init__(self,device="cuda:0",epoch=5,mini_batch_size=256):
        self.device = torch.device(device)
        self.epoch=epoch
        self.mini_batch_size = mini_batch_size

    def train(
            self,
            train_data:EDataSet,
            valid_data:EDataSet,
            model:Energy,
            lr=0.005,
            epsilon=0.05
    ):
        model_path="energy/"
        # for key,value in valid_data.items():
        #     if key=="property":
        #         valid_data[key] = value.to(self.device)
        #     else:
        #         valid_data[key] = value.to(self.device).long()

        epoch_loss = np.array([])
        valid_loss=np.array([])
        optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
        size=train_data.__len__()

        lamb = lambda_lr(warmup_step=math.ceil(size*self.epoch/self.mini_batch_size*0.2),beta=math.log(100)/math.ceil(size*self.epoch/self.mini_batch_size*0.8))

        warmup=torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lamb.lambda_lr)
        loader=MultiEpochsDataLoader(train_data,self.mini_batch_size,shuffle=True,num_workers=4,drop_last=True)
        loader2 = MultiEpochsDataLoader(valid_data,self.mini_batch_size/4,shuffle=True,num_workers=4,drop_last=True)

        for j in range(self.epoch):
            print("----------------epoch: "+str(j+1)+"----------------\n")
            for i,tensors in enumerate(loader):
                x,property=tensors
                num_prop=property.shape[-1]
                energy=model(x,property)
                hard_neg_prop=torch.rand(self.mini_batch_size,num_prop,32,device=self.device)*2-1
                hard_neg_prop=torch.where(hard_neg_prop.abs()<epsilon,torch.zeros_like(hard_neg_prop,device=self.device),hard_neg_prop)
                hard_neg_prop=hard_neg_prop+property.unsqueeze(-1)
                energy_hns=model.hard_neg_energy(x,hard_neg_prop)
                # gc.collect()
                # torch.cuda.empty_cache()
                Loss,sq_loss,exp_loss = E_Loss(
                    energy,
                    energy_hns,
                )
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

                print(
                    f"data:{np.round((i+1)*self.mini_batch_size/x.shape[0]*100,3)}%,loss:{np.round(Loss.item(), 5)}",f"{np.round(sq_loss, 5)},{np.round(exp_loss,5)}")

                epoch_loss = np.append(epoch_loss,(Loss.item(),sq_loss,exp_loss))
            if (j+1)%2==0:
                torch.save(model.state_dict(), folder+model_path+f"reg_model_moses3_epoch{(j+1)}.pth")

            with torch.no_grad():
                for i, tensors in enumerate(loader2):
                    x, property = tensors
                    num_prop = property.shape[-1]
                    energy = model(x, property)
                    hard_neg_prop = torch.rand(self.mini_batch_size, num_prop, 32, device=self.device) * 0.4 - 0.2
                    hard_neg_prop = torch.where(hard_neg_prop.abs() < epsilon,
                                                torch.zeros_like(hard_neg_prop, device=self.device), hard_neg_prop)
                    hard_neg_prop = hard_neg_prop + property.unsqueeze(-1)
                    energy_hns = model.hard_neg_energy(x,hard_neg_prop)
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    Loss, sq_loss, exp_loss = E_Loss(
                        energy,
                        energy_hns,
                    )
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

                    print(
                        f"data:{np.round((i + 1) * self.mini_batch_size / x.shape[0] * 100, 3)}%,loss:{np.round(Loss.item(), 5)}",
                        f"{np.round(sq_loss, 5)},{np.round(exp_loss, 5)}")

                    valid_loss = np.append(valid_loss,(Loss.item(),sq_loss,exp_loss))

        return epoch_loss,valid_loss


def pretrain_model(args):
    loss_path = os.path.join(os.path.abspath(''),'..',"energy_Loss")
    model = Energy(
        num_prop=args.num_prop,
        hidden_dim=args.hidden_dim,
        in_dim=args.in_dim,
        num_block=args.num_block,
        device=args.device
    )
    with open(args.train_path, "rb") as file:
        train_data = pickle.load(file)
    train_data=EDataSet(train_data)
    with open(args.valid_path, "rb") as file:
        valid_data = pickle.load(file)
    valid_data=EDataSet(batch_data=valid_data)
    print("data load done.")
    train=Energy_Train(
        device="cuda:0",
        epoch=20,
        mini_batch_size=256
    )
    # model.load_state_dict(torch.load(folder+"vae/"+"vae_model_zinc6_epoch7.pth"))
    epoch_loss,valid_loss=train.train(
        train_data=train_data,
        valid_data=valid_data,
        model=model,
        lr = 0.005
    )

    pickle.dump(epoch_loss,open(os.path.join(loss_path,"energy_epoch_loss_moses3.pkl"),"wb"))
    pickle.dump(epoch_loss,open(os.path.join(loss_path,"energy_valid_loss_moses3.pkl"),"wb"))
    print("loss save done.")

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_path',default=os.path.join(os.path.abspath(''),'..','MolDataSet',"train_encoding_moses.pkl"))
    parser.add_argument('--valid_path',default=os.path.join(os.path.abspath(''),'..','MolDataSet',"valid_encoding_moses.pkl"))
    parser.add_argument('--batch_size',default=256)
    parser.add_argument('--epoch',default=20)
    parser.add_argument('--num_prop',default=3)
    parser.add_argument('--hidden_dim',default=768)
    parser.add_argument('--in_dim',default= 768)
    parser.add_argument('--num_block',default=3)
    parser.add_argument('--Init_params',default= True)
    parser.add_argument('--device',default="cuda:0")
    args=parser.parse_args()
    pretrain_model(args)
    # finetune_model()