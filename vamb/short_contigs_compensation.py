from argparse import ArgumentError
from sklearn.neighbors import BallTree
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from collections import defaultdict
import pandas as pd


def read_contact_map(contact_map_file, contignames, minscore=None):
    
    minscore = minscore or float("-inf")
    name2id = dict(zip(contignames, range(len(contignames))))
    
    contact_map_raw = pd.read_csv(contact_map_file, sep='\t')[["FirstName", "SecondName", "SpadesScore"]].values
    contacts = defaultdict(lambda: defaultdict(list))
    
    contact_map = tuple(filter(lambda x: x[0] in name2id and x[1] in name2id, contact_map_raw))
    
    for contig_1, contig_2, score in tqdm(contact_map, total=len(contact_map)):
        nameid_1 = name2id[contig_1]
        nameid_2 = name2id[contig_2]
        if score >= minscore:
            contacts[nameid_1][nameid_2] = score
            contacts[nameid_2][nameid_1] = score
    
    contact_map_sorted_by_score = {}
    
    numcontacts = []
    for contigid_i, contacts in contacts.items():
        contact_map_sorted_by_score[contigid_i] = tuple(sorted(contacts.keys(), key=lambda x: contacts[x]))
        numcontacts.append(len(contact_map_sorted_by_score[contigid_i]))
    
    numcontacts = np.array(numcontacts)
    print(f"Mean # of contacts with short contig: {numcontacts.mean()}, STD: {numcontacts.std()}")
    
    return contact_map_sorted_by_score


def normalize_rows_by_length(array):
    x_norm = np.linalg.norm(array, axis=1, keepdims=True)
    x = array / x_norm
    
    return x


def aggregation_step_hic(embeddings, contact_map, contig_lengths, K_neighbours, short_indices, gamma, delta):
    emb_size = embeddings.shape[1]
    new_embeds = []
    
    for short_contigid in tqdm(short_indices):
        top_k_neighborhood_idxs  = contact_map[short_contigid][:min(K_neighbours, len(contact_map[short_contigid]))]
        
        neighbors_lengths = np.take(contig_lengths, top_k_neighborhood_idxs, axis=0)
        neighbors_embeddings = np.take(embeddings, top_k_neighborhood_idxs, axis=0)
        
        overall_length = neighbors_lengths.sum()

        aggregation = np.zeros(emb_size)
        
        for length_k, emb_k in zip(neighbors_lengths, neighbors_embeddings):
            aggregation += length_k / overall_length * emb_k
        
        new_embedding = gamma * embeddings[short_contigid] + delta * aggregation
        
        new_embeds.append(new_embedding)
        
    for i, e in zip(short_indices, new_embeds):
        embeddings[i] = e
        
        
    return embeddings

def aggregate_features(contig_lengths,
                       contact_map,
                       short_indices,
                       gamma,
                       delta,
                       dataloader: torch.utils.data.DataLoader=None,
                       embeddings: np.ndarray=None,
                       K_neighbours=30,
                       TRAINING=True,
                       steps=1,
                       batch_size=None,
                       num_workers=None,
                       pin_memory=None,
                       ):
    
    if TRAINING and dataloader:
        _unchanging_tensor, embeddings = dataloader.dataset[::1]
        embeddings = embeddings.cpu().numpy()
    elif not TRAINING and isinstance(embeddings, np.ndarray):
        embeddings = embeddings
    else:
        raise ArgumentError("Only valid inputs are: Dataloader during training and embeddings np.ndarray during final run!")    
    
    for _ in range(steps):
        embeddings = aggregation_step_hic(embeddings, contact_map, contig_lengths, K_neighbours, short_indices, gamma, delta)
        
    if TRAINING:
        embeddings_tensor = torch.from_numpy(embeddings)
        
        dataset = TensorDataset(_unchanging_tensor, embeddings_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True,
                                shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
        return dataloader
    
    return embeddings

def aggregate_features_old(contig_lengths,
                       short_indices,
                       gamma,
                       delta,
                       dataloader: torch.utils.data.DataLoader=None,
                       embeddings: np.ndarray=None,
                       K_neighbours=30,
                       TRAINING=True,
                       steps=1,
                       batch_size=None,
                       num_workers=None,
                       pin_memory=None,
                       ):
    
    if TRAINING and dataloader:
        _unchanging_tensor, embeddings = dataloader.dataset[::1]
        embeddings = embeddings.cpu().numpy()
    elif not TRAINING and isinstance(embeddings, np.ndarray):
        embeddings = embeddings
    else:
        raise ArgumentError("Only valid inputs are: Dataloader during training and embeddings np.ndarray during final run!")    
    
    for _ in range(steps):
        embeddings = aggregation_step_old(embeddings, contig_lengths, K_neighbours, short_indices, gamma, delta)
        
    if TRAINING:
        embeddings_tensor = torch.from_numpy(embeddings)
        
        dataset = TensorDataset(_unchanging_tensor, embeddings_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True,
                                shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
        return dataloader
    
    return embeddings

def aggregation_step_old(embeddings, contig_lengths, K_neighbours, short_indices, gamma, delta):
    size = embeddings.shape[1]
    
    normalized_embeddings = normalize_rows_by_length(embeddings)
    
    ball_tree = BallTree(normalized_embeddings, leaf_size=1_000)
    
    short_normalized_embeddings = np.take(normalized_embeddings, short_indices, axis=0)
    short_embeddings = np.take(embeddings, short_indices, axis=0)
    
    neighbors_idxs = ball_tree.query(short_normalized_embeddings, k=K_neighbours, return_distance=False)
    
    
    new_embeds = []
    for neighbors_list, short_embedding in tqdm(zip(neighbors_idxs, short_embeddings), total=neighbors_idxs.shape[0]):
        neighbors_lengths = np.take(contig_lengths, neighbors_list, axis=0)
        neighbors_embeddings = np.take(embeddings, neighbors_list, axis=0)
        
        
        overall_sum = neighbors_lengths.sum()
        
        new_embedding = np.zeros(size)
        for length, emb in zip(neighbors_lengths, neighbors_embeddings):
            new_embedding += length / overall_sum * emb
            
        new_embedding *= delta
        new_embedding += gamma * short_embedding
        
        new_embeds.append(new_embedding)
        
    for i, e in zip(short_indices, new_embeds):
        embeddings[i] = e
        
        
    return embeddings