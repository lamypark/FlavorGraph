import random
import operator
import itertools
import numpy as np
from tqdm import tqdm
import networkx as nx

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
        self.rw = False

    def generate_metapaths(self, args):
        """
        generate all possible metapaths
        """
        return_list = []
        if 'CHC' in args.which_metapath:
            print("# Metapath CHC")
            metapath_list = ['compound', 'ingredient+hub']
            return_list.append(metapath_list)

        if 'CHNHC' in args.which_metapath:
            print("# Metapath CHNHC")
            metapath_list = ['compound', 'ingredient+hub', 'ingredient+no_hub', 'ingredient+hub']
            for _ in range(3):
                return_list.append(metapath_list)

        if 'NHCHN' in args.which_metapath:
            print("# Metapath NHCHN")
            metapath_list = ['ingredient+no_hub', 'ingredient+hub', 'compound', 'ingredient+hub']
            return_list.append(metapath_list)

        if 'RandWalk' in args.which_metapath:
            print("# Random Walk Enabled")
            self.rw = True

        if not return_list:
            return None
        else:
            return return_list

    def create_metapath_walks(self, args, num_walks, meta_paths):
        print("## Creating Metapath Walks...")
        walks = []
        for node in tqdm(self.graph.nodes()):
            # num walks (rows)
            for _ in range(num_walks):
                # Random Walk
                if self.rw:
                    walk = self.weighted_small_walk(node)
                    
                # Metapath
                if meta_paths is not None:
                    for meta_path in meta_paths:
                        walk = self.meta_walk(args, node, meta_path)
                        if walk is not None:
                            walks.append(walk)

        print("Number of MetaPath Walks Created: {}".format(len(walks)))
        walks = list(walk for walk,_ in itertools.groupby(sorted(walks)))
        print("Filterd Number of MetaPath Walks: {}".format(len(walks)))

        #print(walks[:10])
        random.shuffle(walks)
        #print(walks[:10])
        print("MetaPath Walks: {}".format(len(walks)))

        file = "{}{}-metapath_{}-whichmeta_{}-num_walks_{}-len_metapath.txt".format(args.input_path, args.idx_metapath, args.which_metapath, args.num_walks, args.len_metapath)
        with open(file, "w") as fw:
            for walk in walks:
                for node in walk:
                    fw.write("{} ".format(node))
                fw.write("\n")

    def meta_walk(self, args, walk_start, meta_path):
        walk_start_info = self.graph.nodes[walk_start]
        walk_start_type = walk_start_info['type']
        walk_start_is_hub = walk_start_info['is_hub']

        if walk_start_type == 'ingredient':
            meta_start = walk_start_type+"+"+walk_start_is_hub
        else:
            meta_start = walk_start_type

        if meta_start != meta_path[0]:
            return None
        else:
            meta_pos = 0
            walk = [walk_start]

            #print("\n")
            #print(meta_path)

            while len(walk) < args.len_metapath:
                meta_pos += 1
                # retreive the neighbors of last walk
                walk_current = walk[-1]
                neighbors = list(nx.neighbors(self.graph, walk_current))
                # if no neighbor, break.
                if len(neighbors) < 1:
                    break
                # filter neighbor according to current meta_path
                meta = meta_path[ meta_pos%len(meta_path) ]
                #print("meta_pos:", meta_pos)
                #print("current meta:", meta)
                filtered_neighbors = self.filter_neighbors(neighbors, meta)
                #print("filtered_neighbors:", filtered_neighbors)

                if len(filtered_neighbors) < 1:
                    break
                walk = walk + [random.sample(filtered_neighbors, 1)[0]]
        #print("complete walk:", walk)

        if len(walk) > 1:
            return walk
        else:
            return None

    def filter_neighbors(self, neighbors, meta):
        filtered = []
        for neighbor in neighbors:
            neighbor_current_info = self.graph.nodes[neighbor]
            neighbor_current_type = neighbor_current_info['type']
            neighbor_current_is_hub = neighbor_current_info['is_hub']

            if neighbor_current_type == 'ingredient':
                neighbor_current_meta = neighbor_current_type+"+"+neighbor_current_is_hub
            else:
                neighbor_current_meta = neighbor_current_type

            if neighbor_current_meta == meta:
                filtered.append(neighbor)
        return filtered

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

class DeepWalker(object):
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

        print("# of DeepWalks: {}".format(len(self.paths)))

        file = "{}{}-deepwalk_{}-num_walks_{}-len_metapath.txt".format(self.args.input_path, self.args.idx_metapath, self.args.number_of_walks, self.args.walk_length)
        with open(file, "w") as fw:
            for walk in self.paths:
                for node in walk:
                    fw.write("{} ".format(node))
                fw.write("\n")

    def learn_base_embedding(self):
        """
        Learning an embedding of nodes in the base graph.
        :return self.embedding: Embedding of nodes in the latent space.
        """
        self.paths = [[str(node) for node in walk] for walk in self.paths]
        model = Word2Vec(self.paths, size = self.args.dimensions, window = self.args.window_size, min_count = 1, sg = 1, workers = self.args.workers, iter = 1)
        self.embedding = np.array([list(model[str(n)]) for n in self.graph.nodes()])
        return self.embedding
