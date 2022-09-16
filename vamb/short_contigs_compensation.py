from argparse import ArgumentError
from sklearn.neighbors import BallTree
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm


def normalize_rows_by_length(array):
    x_norm = np.linalg.norm(array, axis=1, keepdims=True)
    x = array / x_norm
    
    return x


def aggregation_step(embeddings, contig_lengths, K_neighbours, short_indices, gamma, delta):
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


def aggregate_features(contig_lengths,
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
    elif not TRAINING and embeddings:
        embeddings = embeddings
    else:
        raise ArgumentError("Only valid inputs are: Dataloader during training and embeddings np.ndarray during final run!")    
    
    for _ in range(steps):
        embeddings = aggregation_step(embeddings, contig_lengths, K_neighbours, short_indices, gamma, delta)
        
    if TRAINING:
        embeddings_tensor = torch.from_numpy(embeddings)
        
        dataset = TensorDataset(_unchanging_tensor, embeddings_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True,
                                shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
        return dataloader
    
    return embeddings