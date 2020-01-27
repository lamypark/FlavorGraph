import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pickle
import pandas as pd
import numpy as np

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""
def load_augmentive_features(nodes):
    PICKLE_PATH = "./input/node2fp_revised_1120.pickle"
    print("Loading Chemical Vectors from ", PICKLE_PATH)
    with open(PICKLE_PATH, "rb") as handle:
        binary_dict = pickle.load(handle)
    print("Number of Binary Vectors Available: ", len(binary_dict.keys()))
    print("Number of Nodes in Graph: ", len(nodes))
    # assert len(nodes) == len(binary_dict.keys())
    augmentive_matrix = []
    binary_mask = []

    for row_idx in range(len(nodes)):
        binary_vector = binary_dict[int(nodes[row_idx])]
        try:
            binary_vector = list(binary_vector)
            binary_mask.append([1])
        except:
            binary_vector = [0.0 for _ in range(881)]
            binary_mask.append([0])
        augmentive_matrix.append(binary_vector)

    binary_mask = np.array(binary_mask).astype(float)
    augmentive_matrix = np.array(augmentive_matrix).astype(float)
    vector_length = augmentive_matrix.shape[1]
    return torch.tensor(augmentive_matrix, requires_grad=False).float().to("cuda"), vector_length, torch.tensor(binary_mask, requires_grad=False).float().to("cuda")

class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size                # row / 1825
        self.emb_dimension = emb_dimension      # column / 128

        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        self.print_network(self.u_embeddings, "u_embeddings")
        self.print_network(self.v_embeddings, "v_embeddings")

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("\nModel Name: \"{}\"".format(name))
        print(model)
        print("The number of parameters: {}".format(num_params))

    def save_embedding(self, id2word, file_name):
        embed_dict = dict()
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        for wid, w in id2word.items():
            try:
                embed_dict[w] = embedding[wid]
            except:
                print(w)
        with open(file_name, "wb") as handle:
            pickle.dump(embed_dict, handle)


class SkipGramModelAux(SkipGramModel):
    def __init__(self, emb_size, emb_dimension, nodes=None, aux_coef=0.0001, CSP_save=False):
        super(SkipGramModelAux, self).__init__(emb_size, emb_dimension)
        self.emb_size = emb_size                # row / # of vocab size / 8298
        self.emb_dimension = emb_dimension      # column / user-defined vector dimension / 128
        self.aux_coef = aux_coef
        self.aux_loss = 0.0
        self.CSP_save = CSP_save

        self.aug_embeddings, self.aug_dimension, self.binary_masks = load_augmentive_features(nodes)

        # |V| x |d|
        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension)

        # |V| x |d|
        self.v_embeddings = nn.Embedding(self.emb_size, self.emb_dimension)

        # |d| x |881|
        self.encoder = nn.Linear(self.emb_dimension, self.aug_dimension)

        # |d| x |881|
        self.a_embeddings = nn.Embedding(self.emb_size, self.aug_dimension)
        self.a_embeddings.weight = nn.Parameter(self.aug_embeddings, requires_grad=False)

        self.print_network(self.u_embeddings, "u_embeddings")
        self.print_network(self.encoder, "encoder")
        self.print_network(self.a_embeddings, "a_embeddings")

        initrange = 1 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        # nn.init.sparse_(self.u_embeddings.weight.data, sparsity=0.66, std=0.001)
        nn.init.constant_(self.v_embeddings.weight.data, 0)
        # nn.init.sparse_(self.v_embeddings.weight.data, sparsity=0.66, std=0.001)
        nn.init.sparse_(self.encoder.weight.data, sparsity=0.66, std=0.001)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        emb_u = self.encoder(emb_u)
        emb_v = self.encoder(emb_v)
        emb_neg_v = self.encoder(emb_neg_v)

        # For Chemical Structure Prediction Loss
        pos_u_masks = self.binary_masks[pos_u]
        #pos_v_masks = self.binary_masks[pos_v]
        #neg_v_masks = self.binary_masks[neg_v].reshape(-1, 5).reshape(-1)

        aux_emb_u1 = emb_u[pos_u_masks.nonzero()[:, 0]]
        aux_emb_u2 = self.a_embeddings(pos_u)[pos_u_masks.nonzero()[:, 0]]

        #aux_emb_v1 = emb_v[pos_v_masks.nonzero()[:, 0]]
        #aux_emb_v2 = self.a_embeddings(pos_v)[pos_v_masks.nonzero()[:, 0]]

        #aux_neg_v1 = emb_neg_v.reshape(-1, 881)[neg_v_masks.nonzero()[:, 0]]
        #aux_neg_v2 = self.a_embeddings(neg_v).reshape(-1, 881)[neg_v_masks.nonzero()[:, 0]]

        criterion = nn.BCEWithLogitsLoss()

        aux_loss1 = criterion(aux_emb_u1, aux_emb_u2)
        #aux_loss2 = criterion(aux_emb_v1, aux_emb_v2)
        #aux_loss3 = criterion(aux_neg_v1, aux_neg_v2)
        
        #self.aux_loss = aux_loss1 + aux_loss2 + aux_loss3
        self.aux_loss = aux_loss1
        
        # For Main Skip-Gram Loss
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        #return torch.mean(score + neg_score) + (self.aux_coef * self.aux_loss)
        return torch.mean(score + neg_score + (self.aux_coef*self.aux_loss) )


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("\nModel Name: \"{}\"".format(name))
        print(model)
        print("The number of parameters: {}".format(num_params))

    def save_embedding(self, id2word, file_name):
        embed_dict = dict()
        binary_dict = dict()
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        for wid, w in id2word.items():
            try:
                embed_dict[w] = embedding[wid]
            except:
                print(w)
        with open(file_name, "wb") as handle:
            pickle.dump(embed_dict, handle)

        if self.CSP_save:
            transform = self.encoder.weight.cpu().data.numpy()
            for wid, w in id2word.items():
                try:
                    x = np.matmul(transform, embedding[wid])
                    binary_dict[w] = x
                except Exception as e:
                    print(e)
                    print("Something wrong with encoder?", w)

            with open(file_name.replace('.pickle', '_CSPLayer.pickle'), "wb") as handle:
                pickle.dump(binary_dict, handle)
