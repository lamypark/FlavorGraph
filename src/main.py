import torch
from parser import parameter_parser

from utils import tab_printer, graph_reader, evaluate
from dataloader import DataReader, DatasetLoader
from graph2vec import Metapath2Vec
from plotter import plot_embedding

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    2. Metapath2vec with MetaPathWalker - Ingredient-Ingredient / Ingredient-Food-like Compound / Ingredient-Drug-like Compound
    """
    metapath2vec = Metapath2Vec(args, graph)
    metapath2vec.train()

    """
    3. Plot your embedding if you like
    """
    plot_embedding(args, graph)

    """
    4. Evaluate Node Classification & Node Clustering
    """
    evaluate(args, graph)

if __name__ == "__main__":
    main()
