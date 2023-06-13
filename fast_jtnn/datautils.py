import torch
import sys,json
import time
from torch.utils.data import Dataset, DataLoader
from .mol_tree import *
import numpy as np
from .jtnn_enc import JTNNEncoder
from .mpn import MPN
from .jtmpn import JTMPN
import pickle as pickle
import os, random
import torch.utils.data.distributed


class MolTreeFolder(object):

    def __init__(self, data_folder, vocab,prop_path ,batch_size, epoch=0, hvd=True, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        # self.data_files = [os.path.join(tensor_folder,fn) for tensor_folder in os.listdir(data_folder) if 'tensor' in tensor_folder for fn in os.listdir(os.path.join(data_folder,tensor_folder))]
        self.data_files = [i for i in os.listdir(self.data_folder) if '.pkl' in i]
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.batch = []
        # self.ss_info = []
        self.batch_size = batch_size
        self.epoch = epoch
        # ## prop
        self.prop = json.load(open(prop_path,'r'))

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        # batch_to_add = []
        # if self.shuffle:
        #     random.shuffle(self.data_files)
        # ss_info = []
        
        for i, fn in enumerate(self.data_files):
            num_moltrees = 0
            # print(fn)
            f = os.path.join(self.data_folder, fn)
            with open(f, 'rb') as fin:
                fin.seek(0)
                data = pickle.load(fin)
            # num_moltrees = len(data)
            # if batch_to_add != []:
            #     data = np.append(data,batch_to_add)
            # self.batch += data
            # if num_moltrees > 0:
            #     self.ss_info.append([fn.split('.')[0], num_moltrees])
            # if len(self.batch) >= self.batch_size:
            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch
            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) % self.batch_size != 0:
                batch_to_add = batches.pop()
            # else:
            #     batch_to_add = []
            if hvd:
                kwargs = {'num_workers': 1, 
                        'pin_memory': True}#, 'multiprocessing_context': 'forkserver'}
                dataset = MolTreeDataset(batches, self.vocab, self.assm)
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank())
                dataloader = DataLoader(dataset, 
                                        batch_size=1, 
                                        sampler=train_sampler, 
                                        shuffle=False, 
                                        collate_fn=lambda x:x[0], 
                                        **kwargs)
                train_sampler.set_epoch(self.epoch)
            else:
                dataset = MolTreeDataset(batches, self.vocab, self.assm)
                dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x:x[0], num_workers=self.num_workers)
            if dataloader:
                try:
                    for b in dataloader:
                        moltrees = b[0]
                        props = []
                        for moltree in moltrees:
                            try:
                                if self.prop[moltree.smiles]['tot'] >= 5:
                                    props.append(float(self.prop[moltree.smiles]['hs']))
                                else:
                                    props.append(2)
                            except Exception as e :
                                props.append(2)
                                pass
                        # props = [propset[propset['smiles'] == moltree.smiles].iloc[0,1] for moltree in moltrees]
                        yield b + [props]
                    del dataset, dataloader
                except Exception as e:
                    print(('%s failed' %(fn),e))
                
                del data


class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            return tensorize(self.data[idx], self.vocab, assm=self.assm)
        except:
            print(self.data[idx][0].smiles)
            return tensorize(self.data[idx-1], self.vocab, assm=self.assm)
            pass

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = list(zip(*self.data[idx]))
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)


class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [os.path.join(tensor_folder,fn) for tensor_folder in os.listdir(data_folder) if 'tensor' in tensor_folder for fn in os.listdir(os.path.join(data_folder,tensor_folder))]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        # batches_to_add = []
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)
            # data += batches_to_add
            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch
            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            for batch in batches:
                if len(batches) % self.batch_size != 0:
                    batches_to_add = batches.pop()
                else:
                    batches_to_add = []
                dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])#, num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader
 
def tensorize(tree_batch, vocab, assm=True):

    set_batch_nodeID(tree_batch,vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    mpn_holder = MPN.tensorize(smiles_batch)
    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder
    fatoms_final = create_empty_tensor(0, 35)
    fbonds_final = create_empty_tensor(0, 40)
    agraph_final = create_empty_tensor(0, 15).long()
    bgraph_final = create_empty_tensor(0, 15).long()
    scope_final = []
    batch_idx_final = []
    jtmpn_holder_final = [fatoms_final,fbonds_final,agraph_final,bgraph_final,scope_final]
    for i, mol_tree in enumerate(tree_batch):
        if len(mol_tree.smiles) != 1:
            try:
                cands = []
                batch_idx = []
                for node in mol_tree.nodes:
                    if node.is_leaf or len(node.cands) == 1: continue
                    cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
                    batch_idx.extend([i] * len(node.cands))
                jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
                batch_idx_final += batch_idx
                for num in range(len(jtmpn_holder)-1):
                    jtmpn_holder_final[num] = torch.cat((jtmpn_holder_final[num],jtmpn_holder[num]),dim=0)
                for scope in jtmpn_holder[-1]:
                    jtmpn_holder_final[-1].append(scope)
            except:
                pass
    batch_idx_final = torch.LongTensor(batch_idx_final)
    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder_final, batch_idx_final)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
            
def create_empty_tensor(rows, cols):
    return torch.empty(rows, cols)