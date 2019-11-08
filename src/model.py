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
    PICKLE_PATH = "./input/node2fp_revised.pickle"
    with open(PICKLE_PATH, "rb") as handle:
        binary_dict = pickle.load(handle)
    print(len(binary_dict.keys()))
    print(len(nodes))
    augmentive_matrix = []

    for row_idx in range(len(nodes)):
        augmentive_matrix.append(binary_dict[int(nodes[row_idx])])
    augmentive_matrix = np.array(augmentive_matrix).astype(float)
    vector_length = augmentive_matrix.shape[1]
    return torch.tensor(augmentive_matrix, requires_grad=False).float().to("cuda"), vector_length


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, pretrained_weights=None, is_metapath=False):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size                # row / 1825
        self.emb_dimension = emb_dimension      # column / 128
        self.weights = pretrained_weights

        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

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

    def save_embedding(self, id2word, file_name):
        embed_dict = dict()
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        # with open(file_name, 'w') as f:
        #     f.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            try:
                embed_dict[w] = embedding[wid]
                # e = ' '.join(map(lambda x: str(x), embedding[wid]))
                # f.write('%s %s\n' % (w, e))
            except:
                print(w)
        with open(file_name, "wb") as handle:
            pickle.dump(embed_dict, handle)


class SkipGramModelAug(SkipGramModel):
    def __init__(self, emb_size, emb_dimension, pretrained_weights=None, is_metapath=False, nodes=None):
        super(SkipGramModelAug, self).__init__(emb_size, emb_dimension, pretrained_weights, is_metapath)
        self.emb_size = emb_size                # row / 1825
        self.emb_dimension = emb_dimension      # column / 128
        self.weights = pretrained_weights

        self.aug_embeddings, aug_dimension = load_augmentive_features(nodes)

        self.u_embeddings1 = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings1 = nn.Embedding(emb_size, emb_dimension, sparse=True)

        self.u_embeddings2 = nn.Embedding(emb_size, aug_dimension, sparse=True)
        self.v_embeddings2 = nn.Embedding(emb_size, aug_dimension, sparse=True)

        self.u_embeddings2.weight = nn.Parameter(self.aug_embeddings)
        self.v_embeddings2.weight = nn.Parameter(self.aug_embeddings)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = torch.cat((self.u_embeddings1(pos_u), self.u_embeddings2(pos_u)), dim=1)
        emb_v = torch.cat((self.v_embeddings1(pos_v), self.v_embeddings2(pos_v)), dim=1)
        emb_neg_v = torch.cat((self.v_embeddings1(neg_v), self.v_embeddings2(neg_v)), dim=2)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)
