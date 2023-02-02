import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab, MolTree
from .nnutils import create_var, flatten_tensor, avg_pool
from .jtnn_enc import JTNNEncoder
from .jtnn_dec import JTNNDecoder
from .mpn import MPN
from .jtmpn import JTMPN
from torch.utils.data import DataLoader

from .chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
import rdkit
import rdkit.Chem as Chem
import copy, math

class JTProp(nn.Module):
    def __init__(self, hidden_size, latent_size, prop_size):
        super(JTProp, self).__init__()
        self.latent_size = latent_size / 2
        self.hidden_size = hidden_size
        self.prop_size = prop_size
        self.propNN = nn.Sequential(
                # to first layer
                nn.Linear(self.latent_size * 2, self.hidden_size), # to hidden layer
                nn.Tanh(), # activation
                # to second layer
                nn.Linear(self.hidden_size, 64), # to hidden layer
                nn.Tanh(), 
                # to third layer
                nn.Linear(64, 32), # to hidden layer
                nn.Tanh(), 
                # to output
                nn.Linear(32, self.prop_size) # to output
        )
        self.prop_loss = nn.MSELoss()

    def forward(self, z_tree_vecs, z_mol_vecs, prop_batch):

        prop_label_raw = torch.Tensor(prop_batch)
        idxes_to_use = [0 not in label for label in prop_label_raw]
        if True in idxes_to_use:
            prop_label = create_var(prop_label_raw[idxes_to_use])
            all_vec = torch.cat([z_tree_vecs[idxes_to_use], z_mol_vecs[idxes_to_use]], dim=1)
            shape = prop_label.size()
            prop_loss = self.prop_loss(self.propNN(all_vec).reshape(shape), prop_label)
            p_loss = prop_loss.item()
        else:
            prop_loss = torch.Tensor([])
            p_loss = float('nan')

        return prop_loss, p_loss
