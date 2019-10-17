import torch
from parser import parameter_parser
from utils import tab_printer, graph_reader

from metapath2vec import Metapath2Vec
from plotter import plot_graph, plot_embedding

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
    1. read graph
    """
    graph = graph_reader(args.input_nodes, args.input_edges)
    plot_graph(graph)

    """
    2. Simple Node2vec with DeepWalker - Ingredient-Ingredient
    Need to implement
    """

    """
    3. Metapath2vec with MetaPathWalker - Ingredient-Ingredient / Ingredient-Food-like Compound / Ingredient-Drug-like Compound
    """
    #m2v = Metapath2Vec(args, graph)
    #m2v.train()

    #trainer = SplitterTrainer(graph, args, node2ingr)
    #trainer.walk()
    #trainer.split()

    #trainer.base_model_fit(self):
    #trainer.create_split()
    #trainer.fit()
    #trainer.save_embedding()
    #trainer.save_persona_graph_mapping()

if __name__ == "__main__":
    main()
