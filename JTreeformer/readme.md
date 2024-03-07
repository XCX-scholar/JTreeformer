环境:torch>=1.7.0,rdkit>=2023.3.2,pandas任意  
目录说明:  

Criterion:包含测试生成分子质量的文件Criterion.py.  

DataSet:包含根据baseline对数据集进行预处理的文件,gen_gmtransformer_data.py预处理GM-Transformer所用数据集,gen_mrnn_data.py预处理Molecular-RNN所用数据集.JT-VAE和我们的的模型数据集的预处理在其他文件中.  

jtnn_util:包含了从JT-VAE的方法改来的处理分子的文件.chemutils.py包含了分解和集成分子的方法,mol_tree包含生成关节树的方法,tensorize是我们模型数据集的预处理文件.vocab.py包含生成词表的文件.  

Loss:包含损失函数的定义.  

module:包含我们模型的所有模块.其中jtreeformer包含最基础的模块,encoder定义了encoder,decoder定义了decoder,JTreeformer包含了完整的VAE,SkipNet包含了预测噪声所用的网络的定义,DDPM(实际已改为DDIM)包含了扩散和降噪的方法，Regressor包含了利用扩散过程中各个时间步的数据预测改数据还原成分子后性质的网络。

Test:包含了生成分子的各种文件.其中Generate3.py包含了从噪声中采样,以及将噪声解码为分子,tensor2mol_mcts.py包含了Generate3.py所用函数及蒙特卡洛树搜算的方法,plot_attr.py包含了报告里绘图的方法(如需使用,需要matplotlib>=3.5.3,sklearn>=1.0.2,seaborns>=0.12.2).  

Train:包含了训练的文件.其中DDPM_Train.py包含了训练DDIM的方法,vae_train.py包含了训练VAE的方法,regressor_train.py包含了训练回归网络的方法.  

相较最初的模型，现在的模型做出了以下改动：
1.最初代码中的"brother order encoding"代码有误，进行了修正。

2.最初代码中ddim，eta(降噪时随机噪声的比例)固定为0，现在允许自行设置；同时现在允许使用gradient guidance进行条件控制生成。

3.增加了蒙特卡洛树搜索算法，用于从各种异构体中选出优质分子，经测试的确可以提高生产质量并减少生成时间。

4.将关节树集成为分子的算法原本是直接使用"JT_VAE"的代码，经检查后发现作者给出的代码有缺陷，于是进行了修改，修改后能大幅度提升生成分子的unique指标。但IntDiv有一定大下降，需要通过提高ddim降噪步数来弥补。

5.训练VAE时，取消了CLS辅助任务(降低训练难度)，但训练出的模型性能有些许下降。