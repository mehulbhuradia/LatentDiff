import numpy as np
import os
import sys
sys.path.append("../protein_autoencoder/")
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import radius_graph
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from model import ProAuto
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import Data

device = torch.device('cuda:0')

train_set = torch.load(os.path.join('./', 'AFPDB_data_128_Train_complete.pt'))
valid_set = torch.load(os.path.join('./', 'PDB_data_128_Val_complete.pt'))
test_set = torch.load(os.path.join('./', 'PDB_data_128_Test_complete.pt'))
train_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

params = {
    "mp_steps": 4,
    "layers": 2,
    "num_types": 27,
    "type_dim": 32,
    "hidden_dim": 32,
    "out_node_dim": 32,
    "in_edge_dim":32,
    "output_pad_dim": 1,
    "output_res_dim": 20,
    "pooling": True,
    "up_mlp": False,
    "residual": True,
    "noise": False,
    "transpose": True,
    "attn": True,
    "stride": 2, 
    "kernel": 3, 
    "padding": 1
}

model = ProAuto(**params).double().to(device)
###########################################################################
checkpoint = torch.load(<path of protein autoencoder checkpoint>)
model.load_state_dict(checkpoint['model_state_dict'])

def encoder(model, batched_data):
    
    x, coords_ca, edge_index, batch = batched_data.x, batched_data.coords_ca, batched_data.edge_index, batched_data.batch

    h = model.residue_type_embedding(x.squeeze(1).long()).to(device)

    # encoder
    emb_coords_ca, emb_h, batched_data, edge_index = model.encoder(coords_ca, h, edge_index, batch, batched_data)
    
    return emb_coords_ca, emb_h, model.mlp_mu_h(emb_h), model.mlp_sigma_h(emb_h)

train_diffusion_data = []
valid_diffusion_data = []
test_diffusion_data = []
for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
    batch.coords_ca = batch.coords_ca.double()
    batch = batch.to(device)
    with torch.no_grad():
        emb_coords_ca, emb_h, mu_h, sigma_h = encoder(model, batch)
        emb_h = mu_h + torch.exp(sigma_h / 2) * torch.randn_like(mu_h)
        
    
    emb_coords_ca = emb_coords_ca.view(-1, 32, 3)
    emb_h = emb_h.view(-1, 32, 32)
    
    for i in range(emb_h.shape[0]):
        data = Data(coords=emb_coords_ca[i], h=emb_h[i])
        train_diffusion_data.append(data)

for step, batch in enumerate(tqdm(valid_loader, desc="Iteration")):
    batch.coords_ca = batch.coords_ca.double()
    batch = batch.to(device)
    with torch.no_grad():
        emb_coords_ca, emb_h, mu_h, sigma_h = encoder(model, batch)
        emb_h = mu_h + torch.exp(sigma_h / 2) * torch.randn_like(mu_h)

    emb_coords_ca = emb_coords_ca.view(-1, 32, 3)
    emb_h = emb_h.view(-1, 32, 32)
    
    for i in range(emb_h.shape[0]):
        data = Data(coords=emb_coords_ca[i], h=emb_h[i])
        valid_diffusion_data.append(data)


for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
    batch.coords_ca = batch.coords_ca.double()
    batch = batch.to(device)
    with torch.no_grad():
        emb_coords_ca, emb_h, mu_h, sigma_h = encoder(model, batch)
        emb_h = mu_h + torch.exp(sigma_h / 2) * torch.randn_like(mu_h)

    emb_coords_ca = emb_coords_ca.view(-1, 32, 3)
    emb_h = emb_h.view(-1, 32, 32)
    
    for i in range(emb_h.shape[0]):
        data = Data(coords=emb_coords_ca[i], h=emb_h[i])
        test_diffusion_data.append(data)

dataname = "latent_data"
torch.save(train_diffusion_data, f'./{dataname}_train.pt')
torch.save(valid_diffusion_data, f'./{dataname}_val.pt')
torch.save(test_diffusion_data, f'./{dataname}_test.pt')