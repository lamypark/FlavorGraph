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
        self.enable_chemical_walk = False

    def generate_metapaths(self, args):
        """
        generate all possible metapaths
        """
        return_list = []
        if 'HC_CH' in args.which_metapath:
            print("Metapath HC_CH")
            self.enable_chemical_walk = True

        if 'CHNH' in args.which_metapath:
            print("Metapath CHNH")
            metapath_list = ['compound', 'ingredient+hub', 'ingredient+no_hub', 'ingredient+hub']*int(args.len_metapath)
            for _ in range(args.num_metapath):
                return_list.append(metapath_list)

        if 'NHCH' in args.which_metapath:
            print("Metapath NHCH")
            metapath_list = ['ingredient+no_hub', 'ingredient+hub', 'compound', 'ingredient+hub']*int(args.len_metapath)
            for _ in range(args.num_metapath):
                return_list.append(metapath_list)

        if not return_list:
            return None
        else:
            return return_list

    def create_metapath_walks(self, args, num_walks, meta_paths):
        print("\n\nCreating Metapath Walks...")
        walks = []
        for node in tqdm(self.graph.nodes()):
            for _ in range(num_walks):
                if self.enable_chemical_walk:
                    walk = self.chemical_walk(node)
                    if walk is not None:
                        walks.append(walk)

                if meta_paths is not None:
                    for _ in range(num_walks):
                        for meta_path in meta_paths:
                            walk = self.meta_walk(node, meta_path)
                            if walk is not None:
                                walks.append(walk)

        #print("Number of MetaPath Walks Created: {}".format(len(walks)))
        walks = list(walks for walks,_ in itertools.groupby(walks))
        #random.shuffle(walks)
        print("Number of MetaPath Walks Created: {}".format(len(walks)))

        file = "{}metapath_{}-whichmeta_{}-num_walks_{}-len_walk_{}-num_metapath_{}-dim.txt".format(args.input_path_metapaths, args.which_metapath, args.num_walks_metapath, args.len_metapath, args.num_metapath, args.dim)
        with open(file, "w") as fw:
            for walk in walks:
                for node in walk:
                    fw.write("{} ".format(node))
                fw.write("\n")

    def chemical_walk(self, walk_start):
        walk_start_info = self.graph.nodes[walk_start]
        walk_start_type = walk_start_info['type']
        walk_start_is_hub = walk_start_info['is_hub']

        walk = []
        if walk_start_type == 'compound':
            neighbors_chemical = []
            neighbors = list(nx.neighbors(self.graph, walk_start))

            # if too popular compound
            if len(neighbors) > 150 and len(neighbors) < 5:
                #print("Filtered while creating chemical path:", walk_start, walk_start_info['name'], len(neighbors))
                return None

            random.shuffle(neighbors)
            for neighbor in neighbors:
                neighbor_current_info = self.graph.nodes[neighbor]
                neighbor_current_type = neighbor_current_info['type']
                neighbor_current_is_hub = neighbor_current_info['is_hub']
                if neighbor_current_type == 'ingredient' and neighbor_current_is_hub == 'hub':
                    neighbors_chemical.append(neighbor)
            for neighbor in neighbors_chemical:
                walk = walk + [walk_start, neighbor]
            return walk

        elif walk_start_type == 'ingredient' and walk_start_is_hub == 'hub':
            neighbors_chemical = []
            neighbors = list(nx.neighbors(self.graph, walk_start))

            # if less popular ingredient
            if len(neighbors) < 20:
                #print("Filtered while creating chemical path:", walk_start, walk_start_info['name'], len(neighbors))
                return None

            random.shuffle(neighbors)
            for neighbor in neighbors:
                neighbor_current_info = self.graph.nodes[neighbor]
                neighbor_current_type = neighbor_current_info['type']
                if neighbor_current_type == 'compound':
                    neighbors_chemical.append(neighbor)
            for neighbor in neighbors_chemical:
                walk = walk + [walk_start, neighbor]
            return walk
        else:
            return None


    def meta_walk(self, walk_start, meta_path):
        walk_start_info = self.graph.nodes[walk_start]
        walk_start_type = walk_start_info['type']
        walk_start_is_hub = walk_start_info['is_hub']

        if walk_start_type == 'ingredient':
            meta = walk_start_type+"+"+walk_start_is_hub
        else:
            meta = walk_start_type
        meta_pos = 0

        if meta != meta_path[0]:
            return None
        else:
            #print("\n\n")
            #print(meta_path)
            #print(meta)
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

                if len(filtered_neighbors) < 2:
                    break
                walk = walk + [random.sample(filtered_neighbors, 1)[0]]
                #print("\n\n===")
                #print(walk_start)
                #print(meta_path)
                #print("===")
                #print(walk_current)
                #print(neighbors)
                #print(filtered_neighbors)
                #print(walk)

        #return walk
        if len(list(set(walk))) > 5:
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
