import torch
from parser import parameter_parser
from splitter import SplitterTrainer
from utils import tab_printer, graph_reader
from preprocessor import data_reader_small

def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)

    #node2ingr, edge_id2score = data_reader_small(args)
    #graph = graph_reader(node2ingr, edge_id2score)

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
