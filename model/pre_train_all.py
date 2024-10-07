import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
import random
import math
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict
import torch
from mingpt.utils import CfgNode as CN
from tqdm.auto import tqdm
from sklearn.metrics import *
torch.set_default_tensor_type(torch.DoubleTensor)
import geomloss
import scanpy as sc
from torch.utils.data import (DataLoader, Dataset)
import numpy as np
import os

def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(2023)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # print("B, T, C",B, T, C)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # print("q, k, v",q, k, v)
        q = F.relu(q)
        k = F.relu(k)
        v = F.relu(v)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        att = F.relu(att)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 2 * config.n_embd),
            c_proj=nn.Linear(2 * config.n_embd, config.n_embd),
            act=nn.ReLU(),
            dropout=nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class scDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data)
        label = torch.from_numpy(self.label)

        x = torch.tensor(data[idx])
        y = torch.tensor(label[idx])
        return x, y

class GPT(nn.Module):

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.entreg = .1
        C.p = 2
        C.h = 2
        C.loss1 = 50
        C.mod2_dim = 134
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.conf = config
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                                       'gpt-nano': dict(n_layer=1, n_head=config.h, n_embd=config.n_embd),
                                   }[config.model_type])
        print("config.vocab_size, config.n_embd", config.vocab_size, config.n_embd)
        self.pro = nn.Linear(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))  # config.n_embd*config.block_size
        # self.lm_head = nn.Linear(config.n_embd, config.mod2_dim, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.mod2_dim)  # , bias=False
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # # report number of parameters (note we don't count the decoder parameters in lm_head)
        # n_params = sum(p.numel() for p in self.transformer.parameters())
        # print("number of parameters: %.2fM" % (n_params / 1e6,))

    def cross_mod(self, mod1, mod2=None):
        idx = torch.tensor(mod1, dtype=torch.double)
        device = idx.device
        b, t, v = idx.size()
        num_cls = int(self.conf.block_size/2)
        cls1 = torch.zeros(num_cls).long().to(device)
        cls2 = torch.ones(num_cls).long().to(device)
        cls = torch.cat((cls1, cls2), dim=0).to(device)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        cls_emb = self.transformer.wpe(cls) # + cls_emb

        x = self.transformer.drop(idx + pos_emb)
        for block in self.transformer.h:
            x = F.relu(x)
            x = block(x)
        x = F.relu(x)
        x = self.transformer.ln_f(x)
        x = F.relu(x)
        x = torch.mean(x, dim=1)
        emb = x
        logits = self.lm_head(x)
        mod_logits = F.relu(logits)

        # if we are given some desired targets also calculate the loss
        loss = None
        # criterion = nn.CrossEntropyLoss()

        if mod2 is not None:
            targets = torch.tensor(mod2, dtype=torch.double)
            loss1 = F.mse_loss(mod_logits, targets) ** 0.5
            loss = loss1  # loss1 #+ loss2
        return loss, emb,mod_logits

    def forward(self, X, Y):

        loss1, emb_mod,mod1_logits2 = self.cross_mod(X, Y)

        loss = self.conf.loss1*loss1
        return emb_mod, loss,mod1_logits2

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas) #,weight_decay=train_config.weight_decay
        return optimizer

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 1
        # optimizer parameters
        C.epoch = 100
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.95, 0.99)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        model, config = self.model, self.config
        model = model.to(self.device)
        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)#.to(self.device)

        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=False,
                                  pin_memory=True)
        # del train_dataset
        model.train()
        n_epochs = config.epoch
        for epoch in range(n_epochs):

            train_loss = []
            train_loss1 = []
            train_loss3 = []
            emb_mods = []
            emb_mod2s = []
            mod1_logits2s = []
            mod2_logits1s = []
            for batch in tqdm(train_loader):
                X, Y, = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                emb_mod,self.loss,mod1_logits2 = model(X, Y)

                del X,Y,batch

                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                train_loss.append(self.loss.item())
                if(epoch == n_epochs - 1):
                    if(self.device == 'cuda'):
                        emb_mods.extend(emb_mod.cpu().detach().numpy()) #.numpy()
                    else:
                        emb_mods.extend(emb_mod.detach().numpy()) #.numpy()

            train_loss = sum(train_loss) / len(train_loss)


            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f},")

        emb_mods = np.asarray(emb_mods)

        # mod1_logits2s = np.asarray(mod1_logits2s)

        return emb_mods#, mod1_logits2s#, mod2_logits1s


def GeneEmbeding(X, gap):
    num_cells = len(X)
    num_features = (len(X[0]) + gap - 1) // gap
    shape = (num_cells, num_features, gap)

    mmap = np.memmap('gene_embedding.dat', dtype='float32', mode='w+', shape=shape)

    for cell_index, single_cell in enumerate(X):
        # 对于每个single_cell，我们已经知道特征的数量
        for feature_index in range(0, len(single_cell), gap):
            end_index = feature_index + gap
            if end_index > len(single_cell):
                feature = single_cell[-gap:]
            else:
                feature = single_cell[feature_index:end_index]

            mmap[cell_index, feature_index // gap, :] = feature

    mmap.flush()

    print("gene_embedding.dat mmap.shape", mmap.shape)
    return mmap

def getXY2(mod_paths, mod_names, gap):
    # step1: 获得两种模态数据并且进行预处理
    adata_mod1 = sc.read_h5ad(mod_paths[0] + mod_names[0])
    # adata_mod1.var_names_make_unique()
    # adata_mod1.obs['domain_id'] = 0
    # sc.pp.normalize_total(adata_mod1)
    # sc.pp.log1p(adata_mod1)
    # sc.pp.highly_variable_genes(adata_mod1, n_top_genes=2000)
    # adata_mod1 = adata_mod1[:, adata_mod1.var['highly_variable']]


    X1 = adata_mod1.X#obsm['X_pca']#.todense()

    del adata_mod1
    if not isinstance(X1, np.ndarray):
        X1 = X1.todense()

    X1 = np.asarray(X1)

    Y1 = X1

    X1 = GeneEmbeding(X1, gap)

    a,b,c = X1.shape
    num_zeros = int(0.75 * b)
    for i in range(a):
        indices = np.random.choice(b, size=num_zeros, replace=False)
        X1[i, indices, :] = 0

    return X1,Y1


import time

data_name = "Arabidopsis_all_8000_BatchExper"
# test_ADT = "../../Dataset/scglue/“+ data_name + ”/" + data_name
# test_GEX = "../../Dataset/scglue/“ + data_name + ”/" + data_name
path = "../Datasets/Arabidopsis thaliana/All/" +"/"
test_paths = [path]
test_names = [data_name + ".h5ad"]

gap = 128

X1,Y1 = getXY2(test_paths, test_names, gap)
from scipy.sparse import save_npz, load_npz

print("train_mod1.shape", X1.shape)
print("train_mod2.shape", Y1.shape)
mod2_dim =Y1.shape[1]

start = time.time()

a, b, c = X1.shape
print("a,b,c,", a, b, c)
train_dataset = scDataSet(data = X1, label = Y1)
del X1,Y1
model_name = "integration_end_v6"

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = c
model_config.block_size = b
model_config.n_embd = c
model_config.embd_pdrop = 0.0
model_config.resid_pdrop = 0.0
model_config.attn_pdrop = 0.0
model_config.loss1 = 50
model_config.h = 32
model_config.mod2_dim = mod2_dim
print("model_config.mod2_dim ", model_config.mod2_dim)
print("model_config.vocab_size", model_config.vocab_size)
model = GPT(model_config)


train_config = Trainer.get_default_config()
train_config.epoch = 100
train_config.learning_rate = 1e-4
train_config.batch_size = 32  # 10240
trainer = Trainer(train_config, model, train_dataset)
del train_dataset

log_dir = "log/" + str(model_name) + "/"+ str(train_config.epoch) + "/" + str("0.75")+ "/" + data_name + "/"
if (not os.path.isdir(log_dir)):
    os.makedirs(log_dir)

emb_mod1s_ = trainer.run()

end = time.time()
all_time = end - start
print("all_time",all_time)
np.save(log_dir+"emb_mod1s.npy",emb_mod1s_)
torch.save(model.state_dict(), log_dir+'scPlantGPT.pth')



