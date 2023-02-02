import argparse, sys, pickle, os, glob, json, rdkit, socket, math, time
import numpy as np
from re import I
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.multiprocessing as mp
import horovod.torch as hvd
import horovod
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.append('/home/scorej41075/program/JTVAE_horovod/JCCS')
from collections import deque
from fast_jtnn import *
from fast_jtnn.jtprop_vae import JTPropVAE
from tqdm import tqdm
import tempfile
import nvidia_smi
import pandas as pd

t = time.strftime('%Y%m%d%H%M%S')

def main_vae_train(args):
  
    train = args.train
    vocab = args.vocab
    prop_path = args.prop_path
    save_dir = args.save_dir
    load_epoch = args.load_epoch
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    latent_size = args.latent_size
    depthT = args.depthT
    depthG = args.depthG
    lr = args.lr
    clip_norm = args.clip_norm
    beta = args.beta
    step_beta = args.step_beta
    max_beta = args.max_beta
    warmup = args.warmup
    epoch = args.epoch
    anneal_rate = args.anneal_rate
    anneal_iter = args.anneal_iter
    kl_anneal_iter = args.kl_anneal_iter
    print_iter = args.print_iter
    save_iter = args.save_iter
    cuda = args.cuda
    use_adasum = args.use_adasum
    fp16_allreduce = args.fp16_allreduce
    
    fname = args.train.split('/')[-1]
    fout = '%s_JCCS_activatefunc_%s.log' %(fname, t)

    # prop_dict = {'hs':1, 'ls':0}
    # zeff_dct = {'Fe0':0.823899, 'Fe1':0.867925, 'Fe2':0.911950, 'Fe3':0.955975, 'Fe4':1.000000,
    #             'Co0':0.698113, 'Co1':0.742138, 'Co2':0.786164, 'Co3':0.830189, 'Co4':0.874214,
    #             'Mn0':0.572327, 'Mn1':0.616352, 'Mn2':0.660377, 'Mn3':0.704403, 'Mn4':0.748428,'Mn5':0.798742,'Mn6':0.835220,'Mn7':0.880503
    #             }
    hvd.init()

    if cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.empty_cache()

    torch.set_num_threads(1)

    vocab = [x.strip("\n") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTPropVAE(vocab, int(hidden_size), int(latent_size), int(depthT), int(depthG))
    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    
    if load_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/model.epoch-" + str(load_epoch)))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = DDP(model) #, device_ids=[0])#, output_device=0)
    
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = torch.cuda.device_count() if not use_adasum else 1

    if cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    optimizer = optim.Adam(model.parameters(), lr=lr * lr_scaler)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
    scheduler.step()

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                        named_parameters=model.named_parameters(),
                                        compression=compression,
                                        # backward_passes_per_step=10,
                                        op=hvd.Adasum if use_adasum else hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = load_epoch
    beta = beta
    meters = np.zeros(5)
    nancount = 0
    with open(fout,'w') as f:
        s = 'step,Beta,KL,Word,Topo,Assm,PNorm,GNorm,Ploss'
        f.write(s+'\n')
        
    for epoch in tqdm(range(epoch)):
        loader = MolTreeFolder(train, vocab,prop_path, batch_size, epoch=epoch, num_workers=4)
        # print('epoch number is %s' %(epoch))
        for batch in loader:
            # print('batch number is %s' %(total_step))
            # continue
            # props = np.array([[prop_dict[ss], count,zeff_dct[ion]] for ref, count, ss, ion in props]).T
            if cuda:
                batch_new0 = batch[0]
                batch_new1 = batch[1]
                batch_new2 = batch[2]
                batch_new3 = batch[3]
                batch_new4 = torch.Tensor(batch[4])

                batch_new1 = tuple([i.cuda() if ii < 4 else i for ii,i in enumerate(batch_new1)])
                batch_new2 = tuple([i.cuda() if ii < 4 else i for ii,i in enumerate(batch_new2)])
                batch_new30 = batch_new3[0]
                batch_new31 = batch_new3[1]
                batch_new30 = tuple([i.cuda() if ii < 4 else i for ii,i in enumerate(batch_new30)])
                batch_new3 = (batch_new30, batch_new31)
                batch = ((batch_new0, batch_new1, batch_new2, batch_new3,batch_new4))
            # batches.append(batch)
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc, ploss = model(batch, beta)
                # loss, kl_div, sacc, ploss = model(batch, beta)
                total_step += 1
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            except Exception as e:
                print(e)
                continue
            if np.isnan(ploss):
                ploss = 0
                nancount += 1
            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100,ploss])
            
            if total_step % print_iter == 0:
                meters /= print_iter
                with open(fout,'a') as f:
                    s = "%d,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%d" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model),meters[4])
                    f.write(s + '\n')
                nvidia_smi.nvmlInit()
                deviceCount = nvidia_smi.nvmlDeviceGetCount()
                for i in range(deviceCount):
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    print("Device {}: {}, Memory : ({:.2f}% free)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total))
                nvidia_smi.nvmlShutdown()
                i = ["[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model))]            

                sys.stdout.flush()
                meters *= 0
                nancount = 0

            if total_step % save_iter == 0:
                if not os.path.exists(save_dir + "/model.iter-" + str(total_step)):
                    torch.save(model.state_dict(), save_dir + "/model.iter-" + str(total_step))

            if total_step % anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_last_lr()[0])

            if total_step % kl_anneal_iter == 0 and total_step >= warmup:
                beta = min(max_beta, beta + step_beta)
        torch.save(model.state_dict(), save_dir + "/model.epoch-" + str(epoch))

    # cleanup
    
    return model

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--prop_path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--load_epoch', type=int, default=0)

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--step_beta', type=float, default=0.002)
    parser.add_argument('--max_beta', type=float, default=0.016)
    parser.add_argument('--warmup', type=int, default=700)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=460)
    parser.add_argument('--kl_anneal_iter', type=int, default=460)
    parser.add_argument('--print_iter', type=int, default=10)
    parser.add_argument('--save_iter', type=int, default=150)
    
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--use_adasum", default=False)
    parser.add_argument("--fp16_allreduce", default=True)
    parser.add_argument('--num-proc', type=int)
    parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
    parser.add_argument('--communication', default='gloo', help='collaborative communication to use: gloo, mpi')

    args = parser.parse_args()
    print(args)
    
    horovod.run(main_vae_train,
                args=(args,),
                np=args.num_proc,
                hosts=args.hosts,
                use_gloo=args.communication == 'gloo',
                use_mpi=args.communication == 'mpi'
                )