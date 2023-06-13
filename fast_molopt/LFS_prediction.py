import sys,os
import torch
sys.path.append('../')
from rdkit.Chem import PandasTools
import pickle as pickle
from fast_jtnn import *
from fast_jtnn.jtprop_vae import JTPropVAE
from rdkit import RDLogger
import pandas as pd
import itertools
project_root = os.path.dirname(os.path.dirname(__file__))
project_root = '/home/scorej41075/program/LFS_design'
hidden_size = 450
latent_size = 56
depthT = 20
depthG = 3

def check_input(input):
    try:
        val = float(input)
        return val
    except :
        raise ValueError('LFS value must be a number and between 0~1 !')
            
class LFSgenerator():
    def __init__(self,hidden_size=hidden_size,latent_size=latent_size,depthT=depthT,depthG=depthG):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depthT = depthT
        self.depthG = depthG
        self.vocab = os.path.join(project_root,'fast_molopt','data_vocab.txt')
        self.vocab = [x.strip("\r\n ") for x in open(self.vocab)]
        self.vocab = Vocab(self.vocab)
        self._restored = False
    def restore(self, model_path=os.path.join(project_root,'fast_molopt','vae_model','model.epoch-99')):
        model = JTPropVAE(self.vocab, int(self.hidden_size), int(self.latent_size),int(self.depthT), int(self.depthG))
        dict_buffer = torch.load(model_path, map_location='cpu')
        model.load_state_dict(dict_buffer)
        model.eval()
        self._restored = True
        self.model = model
        
    def get_vector(self,smile=''):
        self.restore()
        if not self._restored:
            raise ValueError('Must restore model weights!')
        if not smile:
            return ('')
        smi_target = [smile]
        tree_batch = [MolTree(smi) for smi in smi_target]
        _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model.encode(jtenc_holder, mpn_holder)
        self.z_tree_mean = self.model.T_mean(tree_vecs)
        self.z_mol_mean = self.model.G_mean(mol_vecs)
        self.z_tree_log_var = -torch.abs(self.model.T_var(tree_vecs))
        self.z_mol_log_var = -torch.abs(self.model.G_var(mol_vecs))

    def get_lfs_from_smi(self,smile):
        self.get_vector(smile)
        # z_mol_log_var = -torch.abs(model.G_var(mol_vecs))
        lfs = self.model.propNN(torch.cat((self.z_tree_mean,self.z_mol_mean),dim=1))
        lfs = lfs.item()
        return lfs
    
    def gen_target_smi(self,LFS_target,smile='',step_size=0.1,sign=1):
        LFS_target = check_input(LFS_target)
        if 0 <= LFS_target <= 1:
            flag = True
            while flag:
                smis,zs,ps = [],[],[]
                smis_,pro = [],[]
                count,ploss = 0,1
                lfs = self.get_lfs_from_smi(smile)
                if lfs - LFS_target > 0.3:
                    print('\nWarning! Input smile lfs is %.3f, but target lfs is %s.\nPredict lfs might unconfident!'%(lfs.item(),LFS_target))
                    checkpoint = input('Continue?[y/n]\n')
                    if checkpoint.lower() == 'n':
                        flag = False
                        break
                while ploss > 0.1:
                    if count == 0:
                        epsilon_tree = create_var(torch.randn_like(self.z_tree_mean))
                        epsilon_mol = create_var(torch.randn_like(self.z_mol_mean))
                        z_tree_mean_new = self.z_tree_mean + torch.exp(self.z_tree_log_var / 2) * epsilon_tree * step_size
                        z_mol_mean_new = self.z_mol_mean + torch.exp(self.z_mol_log_var / 2) * epsilon_mol * step_size
                        z_tree_mean = self.z_tree_mean
                        z_mol_mean = self.z_mol_mean
                        count += 1
                    lfs_new = self.model.propNN(torch.cat((z_tree_mean_new, z_mol_mean_new),dim=1))
                    ploss = abs(lfs_new.item() - LFS_target)
                    print(ploss)
                    if ploss > 1:
                        count = 0
                        self.get_lfs_from_smi(smile)
                        zs,ps = [],[]
                        continue
                    delta_tree = sign * step_size * (lfs_new - lfs)/(z_tree_mean_new - z_tree_mean)
                    delta_mol = sign * step_size * (lfs_new - lfs)/(z_mol_mean_new - z_mol_mean)
                    lfs = lfs_new
                    z_tree_mean = (z_tree_mean_new)
                    z_mol_mean = (z_mol_mean_new)
                    z_tree_mean_new = z_tree_mean + delta_tree
                    z_mol_mean_new = z_mol_mean + delta_mol
                zs.append([z_tree_mean,z_mol_mean]) 
                ps.append(lfs_new)
                decode_loss = ''
                smis = [self.model.decode(*z, prob_decode=False) for z in zs]
                for i,j in zip(smis,ps):
                    if i != smile:
                        smis_.append(i)
                        pro.append(j.item())
                        decode_loss = abs(self.get_lfs_from_smi(i) - j.item())
                if smis_ != []:
                    if decode_loss < 0.2:
                        flag = False
                            
            return smis_, pro, decode_loss
        else:
            raise ValueError('target LFS must between 0~1 !')
    

if __name__ == '__main__':
    model = LFSgenerator()
    smis = ['C#N','CC#N']
    # lfs = model.get_lfs_from_smi('C#N')
    tree_batch = []
    for smi in smis:
        mol_tree = MolTree(smi)
        mol_tree.recover()
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        tree_batch.append(mol_tree)
    model.restore()
    
    model_ = model.model
    model_.cuda()

    mol_batch = datautils.tensorize(tree_batch,model.vocab)
    x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = mol_batch
    x_tree_vecs, tree_message, x_mol_vecs = model_.encode(x_jtenc_holder,x_mpn_holder)
    z_tree_vecs, tree_kl = model_.rsample(x_tree_vecs, model_.T_mean, model_.T_var)
    z_mol_vecs, mol_kl = model_.rsample(x_mol_vecs, model_.G_mean, model_.G_var)
    
    
    
    assm_loss, assm_acc = model_.assm(x_batch, x_jtmpn_holder, z_mol_vecs, tree_message)
    jtmpn_holder,batch_idx = x_jtmpn_holder
    fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
    batch_idx = create_var(batch_idx)
    cands_vecs = model_.jtmpn(fatoms,fbonds,agraph,bgraph,scope,tree_message)
    x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)

    x_mol_vecs = model_.A_assm(z_mol_vecs) #bilinear

    


    cand_vecs = model_.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)


