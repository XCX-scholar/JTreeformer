import torch
import torch.nn as nn
import pickle
import math
import numpy as np
import os
import math

from module.Regressor import Regressor
from module.DDPM import *
from Loss.Loss import VAE_Loss3,Reg_Loss
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
class RegrDataSet(Dataset):
    def __init__(self,batch_data):
        self.x = batch_data['x'].long()
        self.property = batch_data['property']
    def __getitem__(self,index):
        return self.x[index],self.property[index]

    def __len__(self):
        return self.x.shape[0]

class Regressor_Train():
    def __init__(self,device="cuda:0",noise_schedule='linear',epoch=5,mini_batch_size=256,num_sample_steps=2000):
        self.device = torch.device(device)
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        if noise_schedule == 'linear':
            self.schedule = linear_beta_schedule
        elif noise_schedule == 'cosine':
            self.schedule = cosine_beta_schedule
        elif noise_schedule == 'quadratic':
            self.schedule = quadratic_beta_schedule
        elif noise_schedule == 'quadratic':
            self.schedule = quadratic_beta_schedule
        elif noise_schedule == 'sigmoid':
            self.schedule = sigmoid_beta_schedule
        self.num_sample_steps = num_sample_steps


        self.betas = cosine_beta_schedule(t=self.num_sample_steps).to(self.device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def diffusion_sampling(self,x_start, times, noise):

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, times, x_start.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, times, x_start.shape).to(self.device)

        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).to(self.device)

    def train(
            self,
            train_data:RegrDataSet,
            valid_data:RegrDataSet,
            model,
            lr=0.005,
    ):
        model_path="regressor/"
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

        for j in range(self.epoch):
            print("----------------epoch: "+str(j+1)+"----------------\n")
            for i,tensors in enumerate(loader):
                x,property=tensors
                times = torch.randint(low=0,high=self.step,size=(x.shape[0],),device=self.device)
                noise = torch.randn_like(x,device=self.device)
                x_t=self.diffusion_sampling(x_start=x, times=times, noise=noise)
                pred_property=model(x_t,times)
                # gc.collect()
                # torch.cuda.empty_cache()
                Loss,w_loss,logp_loss,tpsa_loss = Reg_Loss(
                    property,
                    pred_property
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
                    f"data:{np.round((i+1)*self.mini_batch_size/x.shape[0]*100,3)}%,loss:{np.round(Loss.item(), 5)}",f"{np.round(w_loss, 5)},{np.round(logp_loss, 5)},{np.round(tpsa_loss, 5)}")

                epoch_loss = np.append(epoch_loss,(Loss.item(),w_loss,logp_loss,tpsa_loss))
            if (j+1)%2==0:
                torch.save(model.state_dict(), folder+model_path+f"reg_model_zinc7_epoch{(j+1)}.pth")

            with torch.no_grad():
                loader2=DataLoader(valid_data,self.mini_batch_size/4,shuffle=True,num_workers=4,drop_last=False)
                for i, tensors in enumerate(loader):
                    x, property = tensors
                    times = torch.randint(low=0, high=self.step, size=(x.shape[0],), device=self.device)
                    noise = torch.randn_like(x, device=self.device)
                    x_t = self.diffusion_sampling(x_start=x, times=times, noise=noise)
                    pred_property = model(x_t, times)
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    Loss, w_loss, logp_loss, tpsa_loss = Reg_Loss(
                        property,
                        pred_property
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
                        f"epoch:{j}%,loss:{np.round(Loss.item(), 5)}",
                        f"{np.round(w_loss, 5)},{np.round(logp_loss, 5)},{np.round(tpsa_loss, 5)}")

                    valid_loss = np.append(valid_loss, (Loss.item(), w_loss, logp_loss, tpsa_loss))

        return epoch_loss,valid_loss


def pretrain_model(train_path="train_encoding_zinc7.pkl",valid_path="valid_encoding_zinc7.pkl",batch_size=256,epoch=20):
    loss_path = "vae_loss/"
    model = Regressor(
        in_dim=768,
        out_dim=768,
        hidden_dim=768*2,
        time_embedding_dim=768,
        num_block=3,
        dropout=True,
        dropout_rate=0.2,
        Init_params=True,
        device="cuda:0"
    )
    with open(folder + train_path, "rb") as file:
        train_data = pickle.load(file)
    train_data=RegrDataSet(train_data)
    with open(folder + valid_path, "rb") as file:
        valid_data = pickle.load(file)
    valid_data=RegrDataSet(batch_data=valid_data)
    print("data load done.")
    train=Regressor_Train(
        device="cuda:0",
        noise_schedule='linear',
        epoch=epoch,
        mini_batch_size=batch_size,
        num_sample_steps=2000
    )
    # model.load_state_dict(torch.load(folder+"vae/"+"vae_model_zinc6_epoch7.pth"))
    epoch_loss,valid_loss=train.train(
        train_data=train_data,
        valid_data=valid_data,
        model=model,
        lr = 0.005
    )

    pickle.dump(epoch_loss,open(folder+loss_path+"reg_train_loss_zinc7.pkl","wb"))
    pickle.dump(epoch_loss,open(folder+loss_path+"reg_valid_loss_zinc7.pkl","wb"))
    print("loss save done.")

if __name__=='__main__':
    pretrain_model()
    # finetune_model()
