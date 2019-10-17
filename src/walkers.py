import random
import operator
import itertools
import numpy as np
from tqdm import tqdm
import networkx as nx
from gensim.models import Word2Vec

class MetaPathWalker(object):
    """
    DeepWalk node embedding learner object.
    A barebones implementation of "DeepWalk: Online Learning of Social Representations".
    Paper: https://arxiv.org/abs/1403.6652
    Video: https://www.youtube.com/watch?v=aZNtHJwfIVg
    """
    def __init__(self, args, graph):
        """
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.args = args
        self.graph = graph

    def generate_metapaths(self, args):
        if args.make_meta == 'random':
            return [list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], args.len_metapaths, p=[0.3, 0.25, 0.45])) for _ in range(args.num_metapaths)]
        elif args.make_meta == 'starthub':
            return_list = []
            for _ in range(args.num_metapaths):
                small_list = ['hub+ingredient']
                remain_list = list(np.random.choice(['no_hub+ingredient', 'no_hub+food_comp', 'hub+ingredient'], args.len_metapaths-1, p=[0.3, 0.25, 0.45]))
                return_list.append(small_list + remain_list)
            return return_list
        elif args.make_meta == 'hxxhxxh':
            return_list = []
            for _ in range(args.num_metapaths):
                small_list = ['hub+ingredient']
                remain_list1 = list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], int(args.len_metapaths/2), p=[0.4, 0.4, 0.2]))
                remain_list2 = list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], int(args.len_metapaths/2), p=[0.4, 0.4, 0.2]))
                return_list.append(small_list + remain_list1 + small_list + remain_list2 + small_list)
            return return_list
        else:
            return None

    def create_metapath_walks(self, num_walks, meta_paths):
        print("Creating Metapath Walks...")
        walks = []
        for node in tqdm(self.graph.nodes()):
            node_info = self.graph.nodes[node]
            for meta_path in meta_paths:
                for w in range(num_walks):
                    walk = self.meta_walk(node_info, node, meta_path)
                    if walk is not None:
                        walks.append(walk)

        walks = list(walks for walks,_ in itertools.groupby(walks))
        print("Number of MetaPath Walks Created: {}".format(len(walks)))

        with open("./input/metapaths/metapaths_{}_{}.txt".format(num_walks, len(meta_paths)), "w") as fw:
            for walk in walks:
                for word in walk:
                    fw.write("{} ".format(word_dict[word]))
                fw.write("\n")

    def meta_walk(self, node_info, start_node, meta_path):
        meta_pos = 0
        meta_current = node_info['is_hub']+"+"+node_info['type']

        if meta_current != meta_path[0]:
            return None
        else:
            walk = [start_node]
            while len(walk) < len(meta_path):
                meta_pos += 1
                neighs = list(nx.neighbors(self.graph, walk[-1]))
                if len(neighs) < 2:
                    break
                filtered_neighs = self.filter_neighs(neighs, meta_path[meta_pos], meta_current)
                walk = walk + [random.sample(filtered_neighs, 1)[0]]

        if len(list(set(walk))) == len(meta_path):
            return walk
        else:
            return None

    def filter_neighs(self, neighs, meta_type, meta_current):
        filtered = neighs
        for n in neighs:
            if meta_current != meta_type:
                filtered.remove(n)
        return filtered





class DeepWalker(object):
    """
    DeepWalk node embedding learner object.
    A barebones implementation of "DeepWalk: Online Learning of Social Representations".
    Paper: https://arxiv.org/abs/1403.6652
    Video: https://www.youtube.com/watch?v=aZNtHJwfIVg
    """
    def __init__(self, graph, args):
        """
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args

    def small_walk(self, start_node):
        """
        Doing a truncated random walk.
        :param start_node: Start node for random walk.
        :return walk: Truncated random walk list of nodes with fixed maximal length.
        """
        walk = [start_node]
        while len(walk) < self.args.walk_length:
            if len(list(nx.neighbors(self.graph,walk[-1]))) == 0:
                break
            walk = walk + [random.sample(list(nx.neighbors(self.graph,walk[-1])),1)[0]]
        return walk

    def weighted_small_walk(self, start_node):
        """
        Doing a truncated random walk.
        :param start_node: Start node for random walk.
        :return walk: Truncated random walk list of nodes with fixed maximal length.
        """
        walk = [start_node]
        while len(walk) < self.args.walk_length:
            current_node = walk[-1]
            neighbors_of_end_node = list(nx.neighbors(self.graph,current_node))
            if len(neighbors_of_end_node) == 0:
                break
            next_node = random.sample(neighbors_of_end_node,1)[0]
            walk += [next_node]
        return walk

    def create_features(self):
        """
        Creating random walks from each node.
        """
        self.paths = []
        for node in tqdm(self.graph.nodes()):
            for k in range(self.args.number_of_walks):
                walk = self.weighted_small_walk(node)
                self.paths.append(walk)

    def learn_base_embedding(self):
        """
        Learning an embedding of nodes in the base graph.
        :return self.embedding: Embedding of nodes in the latent space.
        """
        self.paths = [[str(node) for node in walk] for walk in self.paths]
        model = Word2Vec(self.paths, size = self.args.dimensions, window = self.args.window_size, min_count = 1, sg = 1, workers = self.args.workers, iter = 1)
        self.embedding = np.array([list(model[str(n)]) for n in self.graph.nodes()])
        return self.embedding
