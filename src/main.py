import torch
from parser import parameter_parser
from utils import tab_printer, graph_reader
from dataloader import DataReader, DatasetLoader

from graph2vec import Metapath2Vec, Node2Vec
from plotter import plot_embedding

#from splitter import SplitterTrainer
#from preprocessor import data_reader_small

def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)

    """
    1. read graph and load as torch dataset
    """
    graph, graph_ingr_only = graph_reader(args.input_nodes, args.input_edges)
    """
    2. Simple Node2vec with DeepWalker - Ingredient-Ingredient

    self.base_walker = DeepWalker(self.graph, self.args)
    print("\nDoing base random walks.\n")
    self.base_walker.create_features()
    print("\nLearning the base model.\n")
    self.base_node_embedding = self.base_walker.learn_base_embedding()
    print("\nDeleting the base walker.\n")
    del self.base_walker

    """
    #node2vec = Node2Vec(args, graph_ingr_only)
    #node2vec.train()
    #plot_embedding(args, graph)

    """
    3. Metapath2vec with MetaPathWalker - Ingredient-Ingredient / Ingredient-Food-like Compound / Ingredient-Drug-like Compound
    """
    metapath2vec = Metapath2Vec(args, graph)
    metapath2vec.train()

    #plot_embedding(args, graph)

if __name__ == "__main__":
    main()
