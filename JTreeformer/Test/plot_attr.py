import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

folder="path/to/dataset"
filename=["25step_encoding_zinc5.pkl","50step_encoding_zinc5.pkl","125step_encoding_zinc5.pkl","250step_encoding_zinc5.pkl","250step_encoding_zinc5.pkl"]
label=['random','25step','50step','125step','250step','500step']

# end=[0]
# all_data=torch.randn((1000,768),device=torch.device("cuda:0"))
# end.append(1000)
# for name in filename:
#     with open(folder+name,"rb") as file:
#         data=pickle.load(file)
#     end.append(data.shape[0]+end[-1])
#     all_data=torch.cat([all_data,data],dim=0)
#
#
# all_data=all_data.cpu().numpy()
# tsne=TSNE(n_components=2,init='pca',random_state=1)
# print("fittint...")
# all_data=tsne.fit_transform(all_data)
# print("finish.")
# for i in range(1,len(end)):
#     x,y=all_data[end[i-1]:end[i],0],all_data[end[i-1]:end[i],1]
#     plt.scatter(x,y,label=label[i-1],s=5,marker='.')
# plt.title("Distribution of different denoising step")
# plt.legend()
# plt.savefig("D:\\MolDataSet\\denoising_step.png")
# plt.show()

# smiles_name=['FF_random_sample_zinc5.txt','FF_25step_sample_zinc5.txt','FF_50step_sample_zinc5.txt','FF_125step_sample_zinc5.txt','FF_250step_sample_zinc5.txt','FF_500step_sample_zinc5.txt']
# # CO2_index=[]
# for j,name in enumerate(smiles_name):
#     with open(folder +name, 'r') as file:
#         smiles_list=file.readlines()
#     # for i,smiles in enumerate(smiles_list):
#     #     if smiles.strip()=="O=C=O":
#     #         CO2_index.append(j*1000+i)
#
# print(CO2_index)
# all_data=torch.zeros((0,768),device=torch.device("cuda:0"))
# for name in filename:
#     with open(folder+name,"rb") as file:
#         data=pickle.load(file)
#     all_data=torch.cat([all_data,data],dim=0)
#
# all_data=all_data.cpu().numpy()
# size=all_data.shape[0]
# not_co2=[i for i in range(size) if i not in CO2_index]
# tsne=TSNE(n_components=2,init='pca',random_state=1)
# print("fittint...")
# all_data=tsne.fit_transform(all_data)
# print("finish.")
# x,y=all_data[:,0],all_data[:,1]
# plt.scatter(x[not_co2],y[not_co2],label='Others',s=2,marker='.')
# plt.scatter(x[CO2_index],y[CO2_index],label='CO2',s=5,marker='.')
# plt.title("Distribution of latent feature of CO2")
# plt.legend()
# plt.savefig("D:\\MolDataSet\\CO2.png")
# plt.show()

# CO2_index=torch.LongTensor(CO2_index).cuda()
# CO2=all_data[CO2_index]
#
# with open(folder+"train_encoding_zinc5.pkl","rb") as file:
#     train_data=pickle.load(file)
# x=train_data['x']
# train_size=x.shape[0]
# idx=torch.from_numpy(np.random.choice(range(train_size),size=10000,replace=False)).long().cuda()
# x=torch.index_select(x,dim=0,index=idx)
# train_size=10000
# property=train_data['property']
# property=torch.index_select(property,dim=0,index=idx).cpu().numpy()
# w=property[:,0]
# logp=property[:,1]
# tpsa=property[:,2]
# max_w=property[:,0].max()
# max_logp=property[:,1].max()
# max_tpsa=property[:,2].max()
# min_w=property[:,0].min()
# min_logp=property[:,1].min()
# min_tpsa=property[:,2].min()


# colors=list(zip((w-min_w)/(max_w-min_w),(logp-min_logp)/(max_logp-min_logp),(tpsa-min_tpsa)/(max_tpsa-min_tpsa),[1 for i in range(w.shape[0])]))
# all_data=torch.cat([x,CO2])
# all_data=all_data.cpu().numpy()
# tsne=TSNE(n_components=2,init='pca',random_state=1)
# print("fittint...")
# all_data=tsne.fit_transform(all_data)
# print("finish.")
# x,y=all_data[:train_size,0],all_data[:train_size,1]
# plt.scatter(x,y,s=1,marker='.',c=colors)
# x,y=all_data[train_size:,0],all_data[train_size:,1]
# plt.scatter(x,y,s=10,label='CO2')
# plt.title("Distribution of latent feature of CO2")
# plt.legend()
# plt.savefig("D:\\MolDataSet\\3CO2.png")
# plt.show()

smiles_name=['FF_random_sample_zinc5.txt','FF_25step_sample_zinc5.txt','FF_50step_sample_zinc5.txt','FF_125step_sample_zinc5.txt','FF_250step_sample_zinc5.txt','FF_500step_sample_zinc5.txt']
from rdkit.Chem import Descriptors
from rdkit import Chem
from Criterion.Criterion import Validity
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
all_w=[]
all_logp=[]
all_tpsa=[]
for j,name in enumerate(smiles_name):
    with open(folder+name, 'r') as file:
        smiles_list=file.readlines()
    _,valids=Validity(smiles_list)
    w=[]
    logp=[]
    tpsa=[]
    for s in valids:
        mol=Chem.MolFromSmiles(s)
        w.append(Descriptors.MolWt(mol))
        logp.append(Descriptors.MolLogP(mol))
        tpsa.append(Descriptors.TPSA(mol))
    print(j)
    all_w.append(w)
    all_logp.append(logp)
    all_tpsa.append(tpsa)

fig,axs=plt.subplots(3,1)
plt.subplots_adjust(hspace=0.8)

axs[0].set_title("weight")
axs[1].set_title("logP")
axs[2].set_title("TPSA")
for j,name in enumerate(smiles_name):
    sns.kdeplot(all_w[j],ax=axs[0],cut=0,label=label[j])
    # axs[0].legend(label[j+1])
    sns.kdeplot(all_logp[j],ax=axs[1],cut=0,label=label[j])
    # axs[1].legend(label[j+1])
    sns.kdeplot(all_tpsa[j],ax=axs[2],cut=0,label=label[j])
    # axs[2].legend(label[j+1])


axs[0].set_ylabel("")

axs[1].set_ylabel("")

axs[2].set_ylabel("")
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc =[0.6,0.8],prop={'size':10},ncol=2)


plt.savefig("path/to/png")
plt.show()
# for j,name in enumerate(smiles_name):
#     print(label[j+1])