import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import math
import numpy as np

from module.DDPM import GaussianDiffusion
from Loss.Loss import DDPM_Loss
import pickle
import os
from module.jtreeformer_modules import lambda_lr
from torch.utils.data import DataLoader,Dataset


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
class EncodingDataSet(Dataset):
    def __init__(self,batch_data):
        super(EncodingDataSet, self).__init__()
        self.x=batch_data['x'].cpu()

    def __getitem__(self,index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class DDPM_Train():
    def __init__(self,step=2000,device="cuda:7",epoch=10,mini_batch_size=256):
        self.step = step
        self.device = torch.device(device)
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size

    def train(self,train_data:EncodingDataSet,valid_data,model:GaussianDiffusion,lr=0.005):
        model_path = os.path.join(os.path.abspath(''), "ddpm")
        if os.path.isdir(model_path) is False:
            os.makedirs(model_path)
        size=train_data.__len__()
        optimizer = torch.optim.Adam(model.noise_net.parameters(),lr=lr)
        lamb = lambda_lr(warmup_step=math.ceil(size*self.epoch/self.mini_batch_size*0.05),beta=math.log(500)/math.ceil(size*self.epoch/self.mini_batch_size*0.95))
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lamb.lambda_lr)
        epoch_loss = np.array([])
        valid_loss=np.array([])
        loader=DataLoader(train_data,batch_size=256,shuffle=True,num_workers=4,drop_last=True)
        loader2=DataLoader(valid_data,batch_size=256,shuffle=True,num_workers=4,drop_last=True)

        for j in range(self.epoch):

            print("----------------epoch: "+str(j+1)+"----------------\n")
            for i,x in enumerate(loader):

                times = torch.randint(low=0,high=self.step,size=(x.shape[0],),device=self.device)

                noise = torch.randn_like(x,device=self.device)
                x_t = model.diffusion_sampling(x_start=x.to(self.device), times=times,noise=noise)
                predicted_noise = model.noise_net(x_t, times)

                Loss = DDPM_Loss(noise=noise,predicted_noise=predicted_noise,loss_type='l1',device=self.device)

                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()
                warmup.step()

                print(f"data:{round((i+1)*self.mini_batch_size/size*100,3)},loss:{Loss.item()}")

                epoch_loss = np.append(epoch_loss,Loss.item())
            if (j+1)%10==0:
                torch.save(model.state_dict(),os.path.join(model_path,f"l1_ddim_model_moses3_epoch{(j+1)}.pth"))
            with torch.no_grad():
                for i,valid_x in enumerate(loader2):
                    try:
                        times = torch.randint(low=0,high=self.step,size=(valid_x.shape[0],),device=self.device)

                        noise = torch.randn_like(valid_x,device=self.device)
                        x_t = model.diffusion_sampling(x_start=valid_x.to(self.device), times=times,noise=noise)
                        predicted_noise = model.noise_net(x_t, times)

                        Loss = DDPM_Loss(noise=noise,predicted_noise=predicted_noise,loss_type='l1',device=self.device)
                        print(f"epoch:{j},loss:{Loss.item()}")
                        valid_loss = np.append(valid_loss, Loss.item())
                    except:
                        pass

        return epoch_loss,valid_loss

def pretrain_model(args):
    loss_path = os.path.join(os.path.abspath(''),'..',"DDPM_Loss")
    if os.path.isdir(loss_path) is False:
        os.makedirs(loss_path)
    model=GaussianDiffusion(
        latent_space_dim=args.latent_space_dim,
        expand_factor=args.expand_factor,
        time_embedding_dim=args.time_embedding_dim,
        num_block=args.num_block,
        dropout=args.dropout,
        dropout_rate=args.dropout_rate,
        Init_params=args.Init_params,
        noise_schedule=args.noise_schedule,
        num_sample_steps=args.num_sample_steps,
        device=args.device
    )
    print("model establish done.")
    with open(args.train_path, "rb") as file:
        train_data = pickle.load(file)
    train_data=EncodingDataSet(train_data)
    with open(args.valid_path, "rb") as file:
        valid_data = pickle.load(file)
    valid_data=EncodingDataSet(valid_data)
    print("data load done.")
    train=DDPM_Train(step=1000,device=args.device,epoch=args.epoch,mini_batch_size=args.batch_size)
    epoch_loss,valid_loss = train.train(train_data=train_data,valid_data=valid_data,model=model,lr=0.005)
    print("model save done.")
    pickle.dump(epoch_loss,open(os.path.join(loss_path,"ddim_epoch_loss_moses3.pkl"),"wb"))
    pickle.dump(valid_loss,open(os.path.join(loss_path,"ddim_valid_loss_moses3.pkl"),"wb"))
    print("epoch loss save done.")

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_path',default=os.path.join(os.path.abspath(''),'GraphLatentDiffusion3',"MolDataSet","train_encoding_moses3.pkl"))
    parser.add_argument('--valid_path',default=os.path.join(os.path.abspath(''),'GraphLatentDiffusion3',"MolDataSet","valid_encoding_moses3.pkl"))
    parser.add_argument('--batch_size',default=256)
    parser.add_argument('--epoch',default=100)
    parser.add_argument('--latent_space_dim',default= 768)
    parser.add_argument('--expand_factor',default= 4)
    parser.add_argument('--time_embedding_dim',default= 768)
    parser.add_argument('--num_block',default= 6)
    parser.add_argument('--dropout',default= True)
    parser.add_argument('--dropout_rate',default= 0.1)
    parser.add_argument('--Init_params',default= True)
    parser.add_argument('--noise_schedule',default= 'linear')
    parser.add_argument('--num_sample_steps',default= 1000)
    parser.add_argument('--device',default="cuda:7")
    args=parser.parse_args()
    pretrain_model(args)