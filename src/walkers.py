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
        """
        generate all possible metapath
        """
        if args.which_metapath == 'random':
            return [list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], args.len_metapath, p=[0.3, 0.25, 0.45])) for _ in range(args.num_metapath)]

        elif args.which_metapath == 'chem+ii':
            return_list = []
            # 1. ingr-ingr only
            ingr_ingr_list = ['hub+ingredient', 'no_hub+ingredient']*int(args.len_metapath)
            for _ in range(args.num_metapath):
                return_list.append(ingr_ingr_list)
            return return_list

        elif args.which_metapath == 'chem+if':
            return_list = []
            # 1. ingr-ingr only
            ingr_ingr_list = ['hub+ingredient', 'no_hub+ingredient']*int(args.len_metapath)
            for _ in range(args.num_metapath):
                small_list = ['hub+ingredient']
                remain_list = list(np.random.choice(['no_hub+ingredient', 'no_hub+food_comp', 'hub+ingredient'], args.len_metapath-1, p=[0.3, 0.25, 0.45]))
                return_list.append(small_list + remain_list)
            return return_list

        elif args.which_metapath == 'chem+ii+if':
            return_list = []
            # 1. ingr-ingr only
            ingr_ingr_list = ['hub+ingredient', 'no_hub+ingredient']*int(args.len_metapath)
            for _ in range(args.num_metapath):
                small_list = ['hub+ingredient']
                remain_list = list(np.random.choice(['no_hub+ingredient', 'no_hub+food_comp', 'hub+ingredient'], args.len_metapath-1, p=[0.3, 0.25, 0.45]))
                return_list.append(small_list + remain_list)

            # 2. ingr-ingr only
            ingr_ingr_list = ['hub+ingredient', 'no_hub+ingredient']*int(args.len_metapath)
            for _ in range(args.num_metapath):
                return_list.append(ingr_ingr_list)
            return return_list

        else:
            return None

    def create_metapath_walks(self, args, num_walks, meta_paths):
        print("Creating Metapath Walks...")
        walks = []
        for node in tqdm(self.graph.nodes()):
            # only hub-ingr[0] - comp[0] - hub-ingr[0] - comp[1] - ...

            if "chem" in args.which_metapath:
                walk = self.chemical_walk(node)
                if walk is not None:
                    walks.append(walk)

            for _ in range(num_walks):
                for meta_path in meta_paths:
                    walk = self.meta_walk(node, meta_path)
                    if walk is not None:
                        walks.append(walk)

        #print("Number of MetaPath Walks Created: {}".format(len(walks)))
        walks = list(walks for walks,_ in itertools.groupby(walks))
        print("Number of MetaPath Walks Created: {}".format(len(walks)))

        file = "{}metapath_{}-whichmeta_{}-num_walks_{}-len_walk_{}-num_metapath_{}-dim.txt".format(args.input_path_metapaths, args.which_metapath, args.num_walks_metapath, args.len_metapath, args.num_metapath, args.dim)
        with open(file, "w") as fw:
            for walk in walks:
                for node in walk:
                    fw.write("{} ".format(node))
                fw.write("\n")

    def chemical_walk(self, walk_start):
        walk_start_info = self.graph.nodes[walk_start]
        walk_start_ishub = walk_start_info['is_hub']

        walk = []
        if walk_start_ishub == 'hub':
            neighbors_chemical = []
            neighbors = list(nx.neighbors(self.graph, walk_start))
            for neighbor in neighbors:
                neighbor_current_info = self.graph.nodes[neighbor]
                neighbor_current_type = neighbor_current_info['type']
                if neighbor_current_type == 'food_comp' or neighbor_current_type == 'drug_comp':
                    neighbors_chemical.append(neighbor)
            for neighbor in neighbors_chemical:
                walk = walk + [walk_start, neighbor]
            return walk
        else:
            return None


    def meta_walk(self, walk_start, meta_path):
        walk_start_info = self.graph.nodes[walk_start]
        meta = walk_start_info['is_hub']+"+"+walk_start_info['type']
        meta_pos = 0

        # print("\n\n")
        # print(meta_path)
        # print(meta)

        if meta != meta_path[0]:
            return None
        else:
            walk = [walk_start]
            while len(walk) < len(meta_path):
                meta_pos += 1
                # retreive the neighbors of last walk
                walk_current = walk[-1]
                neighbors = list(nx.neighbors(self.graph, walk_current))
                # if only one neighbor, break.
                if len(neighbors) < 2:
                    break
                # filter neighbor according to current meta_path
                filtered_neighbors = self.filter_neighbors(neighbors, meta_path[meta_pos])

                # print("===")
                # print(walk_current)
                # print(neighbors)
                # print(filtered_neighbors)

                if len(filtered_neighbors) < 2:
                    break
                walk = walk + [random.sample(filtered_neighbors, 1)[0]]
                # print(walk)

        if len(list(set(walk))) == len(meta_path):
            return walk
        else:
            return None

    def filter_neighbors(self, neighbors, meta):
        filtered = []
        for neighbor in neighbors:
            neighbor_current_info = self.graph.nodes[neighbor]
            neighbor_current_meta = neighbor_current_info['is_hub']+"+"+neighbor_current_info['type']
            if neighbor_current_meta == meta:
                filtered.append(neighbor)
        return filtered

class DeepWalker(object):
    """
    DeepWalk node embedding learner object.
    A barebones implementation of "DeepWalk: Online Learning of Social Representations".
    Paper: https://arxiv.org/abs/1403.6652
    Video: https://www.youtube.com/watch?v=aZNtHJwfIVg
    """
    def __init__(self, args, graph):
        """
        :param args: Arguments object.
        :param graph: NetworkX graph.
        """
        self.args = args
        self.graph = graph

    def small_walk(self, start_node):
        """
        Doing a truncated random walk.
        :param start_node: Start node for random walk.
        :return walk: Truncated random walk list of nodes with fixed maximal length.
        """
        walk = [start_node]
        while len(walk) < self.args.len_deepwalk:
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
        while len(walk) < self.args.len_deepwalk:
            current_node = walk[-1]
            neighbors_of_end_node = list(nx.neighbors(self.graph,current_node))
            if len(neighbors_of_end_node) == 0:
                break
            next_node = random.sample(neighbors_of_end_node,1)[0]
            walk += [next_node]
        return walk

    def create_deepwalk_paths(self):
        """
        Creating random walks from each node.
        """
        walks = []
        ingrs = []
        for node in tqdm(self.graph.nodes()):
            for k in range(self.args.num_walks_deepwalk):
                walk = self.weighted_small_walk(node)
                walks.append(walk)

        print(len(ingrs))

        walks = list(walks for walks,_ in itertools.groupby(walks))
        print("Number of Deepwalk Walks Created: {}".format(len(walks)))

        file = "{}deepwalk_{}-whichmeta_{}-num_walks_{}-len_walk_{}-dim.txt".format(self.args.input_path_deepwalkpaths, self.args.which_deepwalk, self.args.num_walks_deepwalk, self.args.len_deepwalk, self.args.dim)

        with open(file, "w") as fw:
            for walk in walks:
                for node in walk:
                    node_info = self.graph.nodes[node]
                    fw.write("{} ".format(node))
                fw.write("\n")
