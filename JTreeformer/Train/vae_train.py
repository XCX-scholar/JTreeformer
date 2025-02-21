import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn
import pickle
import os
import math
import argparse

from module.JTreeformer import JTreeformer
from Loss.Loss import VAE_Loss3
from module.jtreeformer_modules import lambda_lr
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm


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
        else:
            return 1

def save_model(model, optimizer, epoch, save_path):
    """Saves the model and optimizer state."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, save_path)
    print(f"Model saved to {save_path}")


def load_model(model, optimizer, load_path):
    """Loads the model and optimizer state."""
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {load_path}, trained for {epoch} epochs.")
    return epoch


def pretrain_model(args):
    """Pretrains the JTreeformer VAE model."""
    device = torch.device(args.device)
    train_dataset = JTDataSet(pickle.load(open(args.train_path, "rb")))
    valid_dataset = JTDataSet(pickle.load(open(args.valid_path, "rb")))
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    valid_loader = MultiEpochsDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    loss1 = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    loss2 = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    model = JTreeformer(
        num_layers_encoder=args.num_layers_encoder,
        num_layers_decoder=args.num_layers_decoder,
        hidden_dim_encoder=args.hidden_dim_encoder,
        expand_dim_encoder=args.expand_dim_encoder,
        hidden_dim_decoder=args.hidden_dim_decoder,
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
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = VAE_Loss3

    start_epoch = 0
    save_dir = os.path.join(os.path.abspath(''), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "jtreeformer_vae.pth")

    if os.path.exists(save_path):
        start_epoch = load_model(model, optimizer, save_path) + 1

    size = train_dataset.__len__()
    lamb = lambda_lr(warmup_step=math.ceil(size // args.batch_size * args.epoch * 1 / 8),
                     beta=math.log(500) / math.ceil(size // args.batch_size * args.epoch * 7 / 8))

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lamb.lambda_lr)
    KL_coef = kl_coef(size=size, batch_size=args.batch_size, epoch=args.epoch)
    train_losses = []
    valid_losses = []
    for epoch in range(start_epoch, args.epoch):
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}") as pbar:
            model.train()
            total_loss = 0
            num_batches = 0
            for batch_data in pbar:
                batch_data_ = dict()
                for k,v in batch_data.items():
                    batch_data_[k] = v.to(device)

                result_node, result_edge, mean_encoding, lnvar_encoding, z = model(batch_data_)

                losses = loss_fn(
                    batch_data=batch_data,
                    result_node=result_node,
                    result_edge=result_edge,
                    mean_encoding=mean_encoding,
                    lnvar_encoding=lnvar_encoding,
                    alpha=args.alpha,
                    loss1=loss1,
                    loss2=loss2,
                    beta=0, # AE performs better than VAE
                    gamma=args.gamma,
                    cls_auxiliary=args.cls_auxiliary,
                    pred_property=model.encoder.property_proj(z) if args.cls_auxiliary else None,
                    device=device)

                loss = losses.get('loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                warmup.step()
                KL_coef.step()

                total_loss += loss.item()
                num_batches += 1
                train_losses.append({k:v.item() for k,v in losses.items()})
                bar_str = "\n".join(f"{k}: {v.item()}" for k, v in losses.items())
                pbar.set_postfix({"loss": bar_str})
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss}")

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{args.epoch} (Validation)") as pbar:
                for batch_data in pbar:
                    batch_data_ = dict()
                    for k, v in batch_data.items():
                        batch_data_[k] = v.to(device)

                    result_node, result_edge, mean_encoding, lnvar_encoding, z = model(batch_data_)

                    losses = loss_fn(
                        batch_data=batch_data,
                        result_node=result_node,
                        result_edge=result_edge,
                        mean_encoding=mean_encoding,
                        lnvar_encoding=lnvar_encoding,
                        alpha=args.alpha,
                        loss1=loss1,
                        loss2=loss2,
                        beta=0,
                        gamma=args.gamma,
                        cls_auxiliary=args.cls_auxiliary,
                        pred_property=model.encoder.property_proj(z) if args.cls_auxiliary else None,
                        device=device)

                    loss = losses.get('loss')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    warmup.step()
                    KL_coef.step()

                    total_loss += loss.item()
                    num_batches += 1
                    train_losses.append({k: v.item() for k, v in losses.items()})
                    bar_str = "\n".join(f"{k}: {v.item()}" for k, v in losses.items())
                    pbar.set_postfix({"val_loss": bar_str})

        avg_val_loss = total_val_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss}")
        save_model(model, optimizer, epoch, save_path)

    with open(os.path.join(os.getcwd(), "training_losses.pkl"), 'wb') as f:
        pickle.dump(losses, f)
    with open(os.path.join(os.getcwd(), "validation_losses.pkl"), 'wb') as f:
        pickle.dump(valid_losses, f)

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
