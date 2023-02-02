import sys
import torch
import math, sys, random, os
sys.path.append('../')
import numpy as np
from rdkit.Chem import PandasTools
import argparse
from collections import deque
import pickle as pickle
from fast_jtnn import *
from fast_jtnn.jtprop_vae import JTPropVAE
import rdkit
from rdkit import RDLogger
from tqdm import tqdm
import pandas as pd
from collections import deque
import tempfile
import os
# control devation
project_root = os.path.dirname(os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--vocab',default=os.path.join(project_root,'fast_molopt','data_vocab.txt'))
parser.add_argument('--model',default=os.path.join(project_root,'fast_molopt','vae_model','model.epoch-99'))
parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)
args = parser.parse_args()
RDLogger.DisableLog('rdApp.*')

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)
model = JTPropVAE(vocab, int(args.hidden_size), int(args.latent_size),int(args.depthT), int(args.depthG))
dict_buffer = torch.load(args.model, map_location='cpu')
model.load_state_dict(dict_buffer)
model.eval()

count = 0
zs = []
ps = []
smi_target = ['CC#N']
step_size = 0.01
p_target = 0
ploss = 10
tree_batch = [MolTree(s) for s in smi_target]
_, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, vocab, assm=False)
tree_vecs, _, mol_vecs = model.encode(jtenc_holder, mpn_holder)
z_tree_mean = model.T_mean(tree_vecs)
z_tree_log_var = -torch.abs(model.T_var(tree_vecs))
z_mol_mean = model.G_mean(mol_vecs)
z_mol_log_var = -torch.abs(model.G_var(mol_vecs))
p = model.propNN(torch.cat((z_tree_mean, z_mol_mean),dim=1))
while ploss > 0.01:
    if count == 0:
        epsilon_tree = create_var(torch.randn_like(z_tree_mean))
        z_tree_mean_new = z_tree_mean + torch.exp(z_tree_log_var / 2) * epsilon_tree * step_size
        epsilon_mol = create_var(torch.randn_like(z_mol_mean))
        z_mol_mean_new = z_tree_mean + torch.exp(z_mol_log_var / 2) * epsilon_mol * step_size
        count += 1
    p_new = model.propNN(torch.cat((z_tree_mean_new, z_mol_mean_new),dim=1))
    ploss = abs(p_new.item() - p_target)
    # if abs(p_new.item() - p_target) < abs(p.item() - p_target):
    #     sign = -1
    # else:
    #     sign = 1
    sign = -1
    delta_tree = sign * step_size * (p_new - p)/(z_tree_mean_new - z_tree_mean)
    delta_mol = sign * step_size * (p_new - p)/(z_mol_mean_new - z_mol_mean)
    z_tree_mean = z_tree_mean_new
    z_mol_mean = z_mol_mean_new
    zs.append([z_tree_mean,z_mol_mean])
    ps.append(p_new)
    z_tree_mean_new = z_tree_mean + delta_tree
    z_mol_mean_new = z_mol_mean + delta_mol
    p = p_new
    print(ploss)
smi = model.decode(z_tree_mean, z_mol_mean, False)
print(smi)
# smis = list(data)
smis = [model.decode(*z, prob_decode=False) for z in zs]
pros = [p.item() for p in ps]
df = pd.DataFrame(np.array([smis,pros]).T)
df.columns = ['smiles', 'prop_pred']
PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'molecules')
htm = df.to_html()
open((os.path.join(os.path.expanduser('~'),'testls2.html')),'w').write(htm)