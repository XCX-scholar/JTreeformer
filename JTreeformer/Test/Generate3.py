import sys
sys.path.append('../')
import torch
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.DDPM import GaussianDiffusion
from module.JTreeformer import JTreeformer
from Test.tensor2mol_mcts2 import *
from jtnn_utils import *
import numpy as np
import pickle
import torch.nn as nn
import time

def Denoising(model: GaussianDiffusion,model2,args):

    sub_timestep=[int((args.end_step-args.start_step)/args.sample_steps)*i for i in range(args.sample_steps)] 
    sub_timestep.reverse()
    predict_sample = torch.randn((args.batch_size, args.latent_space_dim), device=torch.device(args.device))
    for i,time in tqdm(enumerate(sub_timestep), total=args.sample_steps):
        if i<args.sample_steps-1:
            print(time)
            t=torch.full((args.batch_size,),time, dtype=torch.long,device=torch.device(args.device))
            if args.condition:
                pass
            else:
                grad=None
            with torch.no_grad():
                t_minus_one=torch.full((args.batch_size,),sub_timestep[i+1], dtype=torch.long,device=torch.device(args.device))
                predict_sample = model.denoising_sample(predict_sample,t,t_minus_one,eta=args.eta,grad_scale=1 if grad is not None else 0,condition_gradient=grad)
                print(predict_sample.max().item(), predict_sample.min().item())
                # with open(os.path.join(os.path.abspath(''),'encoding',f"moses_encoding_{sub_timestep[i+1]}step.pkl"), "wb") as file:
                #     pickle.dump(predict_sample, file)

    return predict_sample

def Decode(
        args
    ):

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    with open(args.vocab_path, 'r') as file:
        smiles=[x.strip() for x in file.readlines()]
    vocab=Vocab(smiles)
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
        device=args.device,
        feature_test=args.feature_test,
        g_test=args.g_test
    )
    model.encoder.add_module('property_proj', nn.Linear(model.encoder.hidden_dim, 4).to(
        torch.device(args.device)))
    model.load_state_dict(torch.load(args.model_path,map_location=torch.device(args.device)))

    model.decoder.dropout=False


    tgt_file=open(args.store_path, 'a')

    with open(args.laten_encoding_path,"rb") as file:
        latent_encoding=pickle.load(file)
    for i in tqdm(range(latent_encoding.shape[0]),total=latent_encoding.shape[0]):
        e=latent_encoding[i].unsqueeze(0)
        pred_root, pred_nodes=Tensor2MolTree(model,
                   e,
                   vocab,
                   args.max_decode_step,
                   args.prob_decode_node,
                   args.prob_decode_edge,
                   args.device)
        if len(pred_nodes)>0:
            new_smiles=MolTree2Mol(pred_root,pred_nodes)
        else:
            new_smiles="None"
        print(len(pred_nodes), new_smiles)

        tgt_file.write(new_smiles+ '\n')
    tgt_file.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddim_path',default=os.path.join(os.path.abspath(''),'ddpm',"sl1_ddim_model_moses3_epoch20.pth"))
    parser.add_argument('--reg_path', default=os.path.join(os.path.abspath(''),'vae',""))
    parser.add_argument('--batch_size',default= 5000)
    parser.add_argument('--latent_space_dim',default= 768)
    parser.add_argument('--sample_steps',default= 10)
    parser.add_argument('--end_step',default= 1000)
    parser.add_argument('--start_step',default= 0)
    parser.add_argument('--device',default= "cpu")
    parser.add_argument('--condition',default= False)
    parser.add_argument('--eta',default=0)
    args=parser.parse_args()
    model=GaussianDiffusion(
        latent_space_dim=768,
        expand_factor=4,
        time_embedding_dim=768,
        num_block=6,
        dropout=True,
        dropout_rate=0.1,
        Init_params=True,
        noise_schedule='linear',
        num_sample_steps=1000,
        device=args.device
    )

    model.load_state_dict(torch.load(args.ddim_path,map_location=torch.device(args.device)))
    print("load done")
    encoding = Denoising(model=model,model2=None,args=args)
    # pickle.dump(encoding, open(os.path.join(os.path.abspath(''),'encoding',"1000_10_encoding_moses.pkl"), "wb"))

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--laten_encoding_path',default=os.path.join(os.path.abspath(''),'encoding', '1000_10_encoding_moses_ft.pkl'))
#     parser.add_argument('--prob_decode_node',default=False)
#     parser.add_argument('--store_path', default=os.path.join(os.path.abspath(''),'result',"ddim_1000_10_moses.txt"))
#     parser.add_argument('--vocab_path',default=os.path.join(os.path.abspath(''),'GraphLatentDiffusion3','vocab',"vocab_moses.txt"))
#     parser.add_argument('--model_path',default=os.path.join(os.path.abspath(''),'vae',"vae_model_moses_epoch4.pth"))
#     parser.add_argument('--max_decode_step',default=30)
#     parser.add_argument('--prob_decode_edge',default=False)
#     parser.add_argument('--device',default='cpu')
#     parser.add_argument('--feature_test',default=False)
#     parser.add_argument('--g_test',default=False)
#     args=parser.parse_args()

#     Decode(args)
