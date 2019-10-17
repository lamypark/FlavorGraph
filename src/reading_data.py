import numpy as np
import torch
from torch.utils.data import Dataset
from download import AminerDataset
np.random.seed(12345)
import pandas as pd
import networkx as nx
import random
import itertools
from tqdm import tqdm


def filter_neighs(neighs, meta_type, type_dict):
    filtered = neighs
    for n in neighs:
        if type_dict[n] != meta_type:
            filtered.remove(n)
    return filtered

def meta_walk(graph, start_node, meta_path, type_dict):
    meta_pos = 0
    if type_dict[start_node] != meta_path[0]:
        return None
    else:
        walk = [start_node]
        while len(walk) < len(meta_path):
            meta_pos += 1
            if len(nx.neighbors(graph, walk[-1])) < 2:
                break
            neighs = nx.neighbors(graph, walk[-1])
            neighs = filter_neighs(neighs, meta_path[meta_pos], type_dict)
            walk = walk + [random.sample(neighs, 1)[0]]
    if len(list(set(walk))) == len(meta_path):
        return walk
    else:
        return None

def create_metapath_walks(num_walks, meta_paths):
    EDGE_PATH = './food/flavornet_edges_191015.csv'
    TYPE_PATH = './food/flavornet_nodes_191015.csv'
    graph = nx.from_edgelist(pd.read_csv(EDGE_PATH).values.tolist())
    graph.remove_edges_from(graph.selfloop_edges())
    print(list(nx.isolates(graph)))
    
    type_dict = dict()
    word_dict = dict()
    df = pd.read_csv(TYPE_PATH)
    for i in df.index:
        type_dict[df['id'][i]] = '+'.join([df['is_hub'][i], df['node_type'][i]]) if df['node_type'][i] == 'ingredient' else df['node_type'][i]
        word_dict[df['id'][i]] = "@".join([type_dict[df['id'][i]], df['name'][i]])
    walks = []
    for node in tqdm(graph.nodes()):
        for meta_path in meta_paths:
            for w in range(num_walks): 
                walk = meta_walk(graph, node, meta_path, type_dict)
                if walk is not None:
                    walks.append(walk) 
    walks.sort()
    walks = list(walks for walks,_ in itertools.groupby(walks))
    print("Number of MetaPath Walks Created: {}".format(len(walks)))
    
    with open("./food/flavornet_metapaths_{}_{}.txt".format(num_walks, len(meta_paths)), "w") as fw:
        for walk in walks:
            for word in walk:
                fw.write("{} ".format(word_dict[word]))
            fw.write("\n")
class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, download, min_count, care_type, num_walks, meta_paths, skip_meta=False):

        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.download = download
        if not skip_meta:
            create_metapath_walks(num_walks, meta_paths)
        # FB = AminerDataset(self.download)
        self.inputFileName = "./food/flavornet_metapaths_{}_{}.txt".format(num_walks, len(meta_paths))
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="ISO-8859-1"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Metapath2vecDataset(Dataset):
    def __init__(self, data, window_size):
        # read in data, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="ISO-8859-1")

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            pair_catch.append((u, v, self.data.getNegatives(v,5)))
                    return pair_catch


    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)