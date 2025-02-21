import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import torch
from tqdm import tqdm

from module.DDPM import GaussianDiffusion
from Loss.Loss import DDPM_Loss
import pickle
import os
from torch.utils.data import DataLoader, Dataset
import argparse
from module.jtreeformer_modules import lambda_lr
import math

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

class EncodingDataSet(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            batch_data = pickle.load(f)
            self.data = batch_data['x'].cpu()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pretrain_model(args):
    train_dataset = EncodingDataSet(args.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = EncodingDataSet(args.val_data_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model = GaussianDiffusion(
        latent_space_dim=args.latent_space_dim,
        expand_factor=args.expand_factor,
        time_embedding_dim=args.time_embedding_dim,
        num_block=args.num_block,
        dropout=args.dropout,
        dropout_rate=args.dropout_rate,
        noise_schedule=args.noise_schedule,
        num_sample_steps=args.num_sample_steps,
        device=args.device
    ).to(args.device)
    criterion = DDPM_Loss
    size = train_dataset.__len__()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lamb = lambda_lr(warmup_step=math.ceil(size * args.epoch / args.batch_size * 0.05),
                     beta=math.log(500) / math.ceil(size * args.epoch / args.batch_size * 0.95))
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lamb.lambda_lr)
    losses = []
    val_losses = []
    for epoch in range(args.epoch):
        total_loss = 0
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}") as pbar:
            for batch in pbar:
                x_start = batch.to(args.device)
                times = torch.randint(0, model.num_sample_steps, (x_start.shape[0],), device=args.device)
                noise = torch.randn_like(x_start, device=args.device)
                x_noisy = model.diffusion_sampling(x_start, times, noise)
                pred_noise = model.noise_net(x_noisy, times)

                loss = criterion(pred_noise, noise,loss_type=args.loss_type)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                warmup.step()

                total_loss += loss.item()
                losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epoch} (Validation)") as pbar:
                for batch in pbar:
                    x_start = batch['latent'].to(args.device)
                    times = torch.randint(0, model.num_sample_steps, (x_start.shape[0],), device=args.device)
                    noise = torch.randn_like(x_start, device=args.device)
                    x_noisy = model.diffusion_sampling(x_start, times, noise)
                    pred_noise = model.noise_net(x_noisy, times)

                    loss = criterion(pred_noise, noise)
                    total_val_loss += loss.item()
                    pbar.set_postfix({"val_loss": loss.item()})

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss}")

    with open(os.path.join(os.getcwd(), "training_losses.pkl"), 'wb') as f:
        pickle.dump(losses, f)
    with open(os.path.join(os.getcwd(), "validation_losses.pkl"), 'wb') as f:
        pickle.dump(val_losses, f)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), "trained_model.ckpt"))

if __name__ == '__main__':
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
