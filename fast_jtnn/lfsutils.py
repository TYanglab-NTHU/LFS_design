import sys,os, math,json
from rdkit import Chem
import torch
import warnings
from PIL import Image  
from matplotlib import pyplot as plt, ticker
sys.path.append('../')
from rdkit.Chem import MACCSkeys, Draw
from fast_jtnn.vocab import *
from fast_jtnn.nnutils import create_var
from fast_jtnn.datautils import tensorize
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.jtprop_vae import JTPropVAE
from torch.nn import CosineSimilarity
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import imshow, axis,figure
from IPython.display import display
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
hidden_size = 450
latent_size = 56
depthT = 20
depthG = 3
vocab = os.path.join('../','fast_molopt','data_vocab.txt')
vocab = [x.strip("\r\n ") for x in open(vocab)]
vocab = Vocab(vocab)
model_path=os.path.join('../','fast_molopt','vae_model','model.epoch-99')

def check_input(input):
    try:
        val = float(input)
        return val
    except :
        raise ValueError('LFS value must be a number and between 0~1 !')
    
def load_model(vocab=vocab,hidden_size=hidden_size,latent_size=latent_size,depthT=depthT,depthG=depthG):
    model = JTPropVAE(vocab, int(hidden_size), int(latent_size),int(depthT), int(depthG))
    dict_buffer = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(dict_buffer)
    model.eval()
    model.cuda()
    return model
def checksmile(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    smi = Chem.MolToSmiles(mol,kekuleSmiles=True,isomericSmiles=True)
    return smi

def highlight_strength(x):
    df_color = pd.DataFrame('', index=x.index, columns=x.columns)
    df_color['Spectrochemical series'] = ['background-color: red' if val == 'phen' or val == 'N₃⁻' else '' for val in x['']]
    return df_color


class LigandGenerator():
    def __init__(self,vocab=vocab,hidden_size=hidden_size,latent_size=latent_size,depthT=depthT,depthG=depthG):
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depthT = depthT
        self.depthG = depthG
        self.model = load_model(vocab,hidden_size,latent_size,depthT,depthG)
    
    def get_latent(self,ftrain=os.path.join('../fast_molopt/vae_model/latent_train_epoch_99-2'),dim0=0,dim1=1):
        df = pd.read_csv(ftrain,header=None)
        df1 = df.iloc[:,1:]
        scaler = StandardScaler().fit(df1)
        train_data = scaler.transform(df1)
        x_train = train_data[:,dim0]
        y_train = train_data[:,dim1]
        self.x_train = x_train
        self.y_train = y_train
        self.scaler = scaler
        
    def randomgen(self,num=1):
        gensmile = set()
        while len(gensmile) < num:
            z_tree = torch.randn(1, self.latent_size // 2).cuda()
            z_mol = torch.randn(1, self.latent_size  // 2).cuda()
            smi = self.model.decode(z_tree, z_mol, prob_decode=False)
            gensmile.add(smi)
        return gensmile
    
    def get_vector(self,smile=''):
        smi_target = [smile]
        tree_batch = [MolTree(smi) for smi in smi_target]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model.encode(jtenc_holder, mpn_holder)
        z_tree_mean = self.model.T_mean(tree_vecs).cuda()
        z_mol_mean = self.model.G_mean(mol_vecs).cuda()
        z_tree_log_var = -torch.abs(self.model.T_var(tree_vecs)).cuda()
        z_mol_log_var = -torch.abs(self.model.G_var(mol_vecs)).cuda()
        return z_tree_mean,z_mol_mean,z_tree_log_var,z_mol_log_var
    
    def get_lfs_from_smi(self,smile):
        z_tree_mean,z_mol_mean,z_tree_log_var,z_mol_log_var = self.get_vector(smile)
        lfs_pred = self.model.propNN(torch.cat((z_tree_mean.cuda(),z_mol_mean.cuda()),dim=1))
        lfs_pred = torch.clamp(lfs_pred,min=0,max=1)
        lfs = lfs_pred.item()
        return lfs
    
    def gen_from_target_withoutprop(self,target_smile,numsmiles=5,step_size=0.01):
        if not target_smile:
            raise ValueError('Target smile not defined! Change to random seed')
        decode_smiles_set = set()
        target_smile_cheeck = checksmile(target_smile)
        z_tree_mean,z_mol_mean,tree_var,mol_var = self.get_vector(target_smile)
        count = 0
        while len(decode_smiles_set) < numsmiles :
            epsilon_tree = create_var(torch.randn_like(z_tree_mean))
            epsilon_mol = create_var(torch.randn_like(z_mol_mean))
            z_tree_mean_new = z_tree_mean + torch.exp(tree_var / 2) * epsilon_tree * step_size
            z_mol_mean_new = z_mol_mean + torch.exp(mol_var / 2) * epsilon_mol * step_size
            smi = self.model.decode(z_tree_mean_new, z_mol_mean_new, prob_decode=False)
            count += 1
            if smi not in decode_smiles_set and smi != target_smile_cheeck:
                decode_smiles_set.add(smi)
            if count > numsmiles :
                step_size += 0.01
                count = 0
        return decode_smiles_set
    

    def LFS_metric(self):
        data = [
        ['H\u2082O', '0.943', 'Weak'],
        ['Cl\u207B', '0.917', 'Weak'],
        ['OH\u207B', '0.889', 'Weak'],
        ['phen', '0.85', 'Strong'],
        ['\u2013N=C=S', '0.786', 'Weak'],
        ['F\u207B', '0.528', 'Weak'],
        ['MeCN', '0.484', 'Intermediate'],
        ['Py', '0.639', 'Intermediate'],
        ['en', '0.429', 'Strong'],
        ['N\u2083\u207B', '0.420', 'Weak'],
        ['CO', '0.202', 'Strong'],
        ['NH\u2083', '0.077', 'Strong'],
        ['\u2013NO\u2082', '0.045', 'Strong'],
        ['PPh\u2083', '0.029', 'Strong'],
        ['CN\u207B', '0', 'Strong']
        ]
        df = pd.DataFrame(data, columns=['', 'LFS metric', 'Spectrochemical series'])
        styled_df = df.style.apply(highlight_strength, axis=None)
        return styled_df
    
    def show_parity_plot(self,df):
        y1_lfs = df['lfs_pred']
        x1_lfs = df['lfs_true']  
        lfs_rmse1 = np.sqrt(np.mean((x1_lfs-y1_lfs)**2))
        lfs_mae_1 = np.mean(abs(x1_lfs-y1_lfs))
        fig, ax = plt.subplots(figsize=(6,6))
        # print(f)
        target = 'lfs'
        xmin, xmax = -0.005,1.005
        ymin, ymax = -0.005,1.005
        xmajor, xminor = 0.25,0.125
        ymajor, yminor = 0.25,0.125
        xlabel, ylabel = '%s$_{true}$'%(str.upper(target)),'%s$_{predict}$' %(str.upper(target))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(xmajor))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ymajor))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(yminor))
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ## find the boundaries of X and Y values
        # Ensure the aspect ratio is square
        # ax.set_aspect("equal", adjustable="box")
        # rmse2 = np.sqrt(np.mean((x2-y2)**2))
        # mae_2 = np.mean(abs(x2-y2))

        dist = [[0,5],[0,5]]
        #string = '\n'.join(('train RMSE %.3f'%rmse1)),'test RMSE %.3f'%rmse2))
        string = 'train RMSE %.3f \ntrain MAE   %.3f'%(lfs_rmse1,lfs_mae_1)#%(rmse1,rmse2,mae_1,mae_2)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, string, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',bbox=props)
        ax.scatter(x1_lfs,y1_lfs,c='royalblue',edgecolors='black',label='train')
        # ax.scatter(x2,y2,c='red',edgecolors='black',label='test')
        ax.plot([0, 1], [0, 1], "--",lw=2,c = 'black',transform=ax.transAxes)        

    def show_model_architecture(self):
        image_path = '../data/latent/model_architecture.png'
        pil_im = Image.open(image_path)

        # Adjust figure size and DPI
        fig = figure(figsize=(3, 3), dpi=400)
        ax = fig.add_subplot(111)
        
        # Display image
        ax.imshow(pil_im)
        axis('off')

    def scatter_plot(self,df_all,dim1=0, dim2=1):
        df_all = np.array(df_all).T
        for idx,df_ in enumerate(df_all):
            smi = df_[0]
            p = df_[2]
            mol = Chem.MolFromSmiles(smi)
            img = Draw.MolsToGridImage([mol],molsPerRow=1)
            png = img.data
            with open(os.path.join('../','data','%s.png'%smi),'wb+') as outf:
                outf.write(png)
            image = Image.open(os.path.join('../','data','%s.png'%smi))
            fig, axes = plt.subplots(1,2, figsize=(8, 6))  # (rows, columns, figsize)
            self.get_latent()
            xmin, xmax = -5, 5
            ymin, ymax = -5, 5
            xmajor, xminor = 2.5, 1.25
            ymajor, yminor = 2.5, 1.25
            if idx != 0:
                df_old = pd.DataFrame(df_all[idx-1][1])
                train_data_old = self.scaler.transform(df_old)
            df_new = pd.DataFrame(df_all[idx][1])
            train_data = self.scaler.transform(df_new)
            xlabel = 'Z(%s)' % dim1
            ylabel = 'Z(%s)' % dim2
            axes[0].set_position([0.1, 0.1,0.8, 1.0])  # [left, bottom, width, height]
            axes[1].set_position([1, 0.5, 0.3, 0.4])  # [left, bottom, width, height]
            axes[1].imshow(image)  # 第二個子圖的資料
            axes[1].text(0.5, 1.0, '%.2f' %(p), horizontalalignment='center', verticalalignment='bottom', transform=axes[1].transAxes, fontsize=16)
            # for i in ['top', 'bottom', 'left', 'right']:
            #     axes[1].spines[i].set_visible(0)
            axes[1].axis('off')
            axes[0].scatter(self.x_train, self.y_train, c='black', s=0.5, edgecolors='black', alpha=0.2)
            if idx != 0:
                axes[0].scatter(train_data_old[0][dim1], train_data_old[0][dim2], c='grey', s=20, edgecolors='black', alpha=1)
            axes[0].scatter(train_data[0][dim1],train_data[0][dim2],c='blue', s=30, edgecolors='black', alpha=1)

            axes[0].xaxis.set_major_locator(ticker.MultipleLocator(xmajor))
            axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
            axes[0].yaxis.set_major_locator(ticker.MultipleLocator(ymajor))
            axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(yminor))
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            axes[0].set_xlim(xmin, xmax)
            axes[0].set_ylim(ymin, ymax)
            axes[0].set_xlabel(xlabel, fontsize=16)
            axes[0].set_ylabel(ylabel, fontsize=16)
            plt.show()
        
    def density_plot(self):
        num = [0,40,99]
        files = ['../data/latent/withlfs_vecs_epoch_%s' %i for i in num]
        files += ['../data/latent/Without_lfs_vecs_epoch_%s' %i for i in num]
        dim1 = 0
        dim2 = 1
        fig, axs = plt.subplots(2,3, figsize=(20, 14))
        axs = axs.flatten()

        for i, file in enumerate(files):
            epoch = file.split('_')[-1]
            df = pd.read_csv(file, header=0)
            df1 = df.iloc[:, 1:]
            scaler = StandardScaler().fit(df1)
            train_data = scaler.transform(df1)
            x_pca_train = train_data
            x_train = x_pca_train[:, dim1]
            y_train = x_pca_train[:, dim2]
            xy = np.vstack([x_pca_train[:,dim1],x_pca_train[:,dim2]])
            z_train = gaussian_kde(xy)(xy)
            idx = z_train.argsort()
            x_train = x_pca_train[:,dim1]
            y_train = x_pca_train[:,dim2]
            x_train = x_train[idx]
            y_train = y_train[idx]
            z_train = z_train[idx]
            xmin, xmax = -5, 5
            ymin, ymax = -5, 5
            xmajor, xminor = 2.5, 1.25
            ymajor, yminor = 2.5, 1.25

            ax = axs[i]
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xmajor))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ymajor))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(yminor))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            mappable = ax.scatter(x_train,y_train,c=z_train,s=1,cmap='Spectral')

            xlabel = 'Z(%s)' % dim1
            ylabel = 'Z(%s)' % dim2
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            if i <= 2:
                ax.text(0.05, 0.9, 'Epoch-%s-withLFS' % epoch, transform=ax.transAxes, fontsize=14, fontweight='bold')
            else:
                ax.text(0.05, 0.9, 'Epoch-%s-withoutLFS' % epoch, transform=ax.transAxes, fontsize=14, fontweight='bold')


        plt.tight_layout()
        plt.show()
    
    def LFS_densityplot(self):
        prop = json.load(open('../data/latent/pross.json','r'))
        num = [0,40,99]
        files = ['../data/latent/withlfs_vecs_epoch_%s' %i for i in num]
        files += ['../data/latent/Without_lfs_vecs_epoch_%s' %i for i in num]
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        for i, file in enumerate(files):
            epoch = file.split('_')[-1]
            df = pd.read_csv(file, header=0)
            df1 = df.iloc[:, 1:]
            data = []
            count = []
            color = []

            for j, mol_id in enumerate(df.iloc[:, 0]):
                try:
                    if prop[mol_id]['tot'] >= 5:
                        data.append(prop[mol_id]['hs'])
                        count.append(j)
                except Exception as e:
                    pass

            scaler = StandardScaler().fit(df1)
            train_data = scaler.transform(df1)
            x_test = train_data[count]
            dim1 = 0
            dim2 = 5
            x_train = train_data[:, dim1]
            y_train = train_data[:, dim2]
            xy = np.vstack([train_data[:,dim1],train_data[:,dim2]])
            z_train = gaussian_kde(xy)(xy)
            idx = z_train.argsort()
            x_train = train_data[:,dim1]
            y_train = train_data[:,dim2]
            x_train = x_train[idx]
            y_train = y_train[idx]
            z_train = z_train[idx]

            for d in data:
                color.append(float(d))

            xmin, xmax = -5, 5
            ymin, ymax = -5, 5
            xmajor, xminor = 2.5, 1.25
            ymajor, yminor = 2.5, 1.25

            ax = axes.flatten()[i]  # Select the current subplot

            # Set major and minor tick locators
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xmajor))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ymajor))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(yminor))

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.tricontour(x_train, y_train, z_train,colors='k',alpha=0.3,linestyles='dotted')

            ax.scatter(x_train, y_train, c='black', s=0.5, edgecolors='black', alpha=0.2)
            mappable = ax.scatter(x_test[:, dim1], x_test[:, dim2], c=color, cmap='Spectral', s=5)

            ax.set_xlabel('Z({})'.format(dim1), fontsize=16)
            ax.set_ylabel('Z({})'.format(dim2), fontsize=16)
            if i <= 2:
                ax.text(0.05, 0.9, 'Epoch-%s-withLFS' % epoch, transform=ax.transAxes, fontsize=20, fontweight='bold')
            else:
                ax.text(0.05, 0.9, 'Epoch-%s-withoutLFS' % epoch, transform=ax.transAxes, fontsize=20, fontweight='bold')

            # Create a colorbar for the current subplot
        cax = fig.add_axes([0.92 + (i * 0.03), 0.15, 0.02, 0.7])  # Adjust the position and size of the colorbar
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label('LFS value', size=15)

        plt.tight_layout()
        plt.show()
        
    def LFS_optimization(self,LFS_target,inputsmile='',step_size=0.1,sign=-1,max_cycle=100,train_file='../data/latent_train_epoch_99-2.csv'):
        print('Running optimizaiotn...')
        # cos = CosineSimilarity(dim=1)
        df = pd.read_csv(train_file,header=None)
        zs_ref = torch.Tensor(df.iloc[:,1:].values).cuda()
        LFS_target = check_input(LFS_target)
        inputsmicheck = checksmile(inputsmile)
        # output = []
        if 0 <= LFS_target <= 1:
            flag = True
            while flag:
                smis,zs,ps = [],[],[]
                count,ploss = 0,1.0
                ploss_threshold = 0.05
                # lfs = self.get_lfs_from_smi(inputsmile)
                t = 0
                while ploss > ploss_threshold and not math.isnan(ploss):
                    if count == 0:
                        z_tree_mean,z_mol_mean,tree_var,mol_var = self.get_vector(inputsmile)
                        epsilon_tree = create_var(torch.randn_like(tree_var))
                        epsilon_mol = create_var(torch.randn_like(mol_var))
                        lfs = self.model.propNN(torch.cat((z_tree_mean, z_mol_mean),dim=1))
                        lfs = torch.clamp(lfs,min=0,max=1).item()
                        delta_tree = torch.exp(tree_var / 2) * epsilon_tree * step_size
                        delta_mol = torch.exp(mol_var / 2) * epsilon_mol * step_size
                        z_tree_mean_new = z_tree_mean + delta_tree
                        z_mol_mean_new = z_mol_mean + delta_mol
                        count += 1
                    lfs_new = self.model.propNN(torch.cat((z_tree_mean_new, z_mol_mean_new),dim=1))
                    lfs_new = torch.clamp(lfs_new,min=0,max=1).item()
                    ploss = abs(lfs_new - LFS_target)
                    delta_tree = sign * step_size * 2 * (lfs_new - LFS_target) * (lfs_new - lfs) / delta_tree / torch.sqrt(torch.Tensor([t+1]).cuda()) 
                    delta_mol = sign * step_size * 2 * (lfs_new - LFS_target) * (lfs_new - lfs) / delta_mol / torch.sqrt(torch.Tensor([t+1]).cuda()) 
                    # delta_tree = sign * step_size * ((lfs_new - lfs) / (z_tree_mean_new - z_tree_mean) * (1 + 2 * (lfs_new - LFS_target)) + 2 * (lfs_new - LFS_target) * (lfs - LFS_target) / (z_tree_mean_new - z_tree_mean)) # / torch.sqrt(torch.Tensor([t+1])) 
                    # delta_mol = sign * step_size * ((lfs_new - lfs) / (z_mol_mean_new - z_mol_mean) * (1 + 2 * (lfs_new - LFS_target)) + 2 * (lfs_new - LFS_target) * (lfs - LFS_target) / (z_mol_mean_new - z_mol_mean)) # / torch.sqrt(torch.Tensor([t+1])) 
                    # print(delta_tree[0])
                    lfs = lfs_new
                    z_tree_mean = (z_tree_mean_new)
                    z_mol_mean = (z_mol_mean_new)
                    z_tree_mean_new = z_tree_mean + delta_tree
                    z_mol_mean_new = z_mol_mean + delta_mol
                    t += 1
                    count += 1
                    zs.append((z_tree_mean,z_mol_mean))
                    ps.append(lfs)
                    if len(ps) > max_cycle or math.isnan(ploss):
                        step_size += 0.01
                        count,ploss = 0,1 
                        zs,ps = [],[]
                          
                if ps != []:
                    if (ps[-1] - LFS_target) < ploss_threshold:
                        print('Start decoding...')
                        smis = [self.model.decode(*z, prob_decode=False) for z in zs]
                        smis_uniq = []
                        idxes = []
                        for i, smi in enumerate(smis[::-1]):
                            if smi not in smis_uniq and smi != inputsmicheck:
                                smis_uniq.append(smi)
                                idxes.append(len(smis)-1-i)
                        if len(smis_uniq) > 1 and inputsmile not in smis_uniq:
                            # yield (smi,torch.cat((z_tree_mean,z_mol_mean),dim=1).tolist())
                            zs = [torch.cat(z,dim=1).tolist() for z in zs]
                            zs = [zs[idx] for idx in idxes[::-1]]
                            smis = smis_uniq[::-1]
                            ps = [ps[idx] for idx in idxes[::-1]]
                            return smis, zs, ps
                        elif inputsmile in smis_uniq:
                            zs,ps = [],[]
                            count,ploss = 0,1 
                            print('Warning! Input smiles is the output smiles!')
                        elif None in smis_uniq:
                            zs,ps = [],[]
                            count,ploss = 0,1 
                            print('Output smiles failure')
                        else:
                            zs,ps = [],[]
                            count,ploss = 0,1 
                            print('There are fewer than 2 molecules found. Restarting...')
                else:
                    count,ploss = 0, 1
                    zs,ps = [],[]

        else:
            raise ValueError('target LFS must between 0~1 !')

if __name__ == '__main__':
    generator = LigandGenerator()
    lfs_optimizaiotn = generator.LFS_optimization(0.4,'C#N')
    for opt in lfs_optimizaiotn:
        print(opt)
