import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="FlavorNet2.0")

    # Input-Output
    parser.add_argument('--input_nodes',
                        default="./input/nodes_191016.csv",
                        type=str, help="input_file")
    parser.add_argument('--input_edges',
                        default="./input/edges_191016.csv",
                        type=str, help="input_file")
    parser.add_argument('--input_path_metapaths',
                        default="./input/metapaths/",
                        type=str, help="input_path_metapaths")

    parser.add_argument('--output_path',
                        default="./output/",
                        type=str, help="output_path")


    # Skip-Gram
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--window_size', default=7, type=int, help="context window size")
    parser.add_argument('--iterations', default=5, type=int, help="iterations")
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    parser.add_argument('--care_type', default=0, type=int, help="if 1, heterogeneous negative sampling, else normal negative sampling")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--min_count', default=5, type=int, help="min count")
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")

    # Node2vec - DeepWalker

    # Metapath2vec - MetapathWalker
    parser.add_argument('--num_walks', default=10, type=int, help="number of walks")
    parser.add_argument('--num_metapaths', default=10, type=int, help="number of metapaths")
    parser.add_argument('--len_metapaths', default=10, type=int, help="length of metapaths")
    parser.add_argument('--make_meta', default='starthub', type=str)
    parser.add_argument('--skip_meta', default=False, action="store_true")

    # Pytorch
    parser.add_argument("--seed",
                        type = int,
                        default = 42, help = "Random seed for PyTorch. Default is 42.")

    """
    parser = argparse.ArgumentParser(description = "Run Splitter.")

    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/edge_id2score.csv",
	                help = "edges csv.")

    parser.add_argument("--data-path",
                        nargs = "?",
                        default = "./input/D1_im2recipe_pairing_scores.csv",
	                help = "all pairing scores csv.")

    parser.add_argument("--embedding-output-path",
                        nargs = "?",
                        default = "./output/embedding.csv",
	                help = "Embedding output path.")

    parser.add_argument("--persona-output-path",
                        nargs = "?",
                        default = "./output/personas.json",
	                help = "Persona output path.")

    parser.add_argument("--number-of-walks",
                        type = int,
                        default = 10,
	                help = "Number of random walks per source node. Default is 10.")

    parser.add_argument("--window-size",
                        type = int,
                        default = 5,
	                help = "Skip-gram window size. Default is 5.")

    parser.add_argument("--negative-samples",
                        type = int,
                        default = 5,
	                help = "Negative sample number. Default is 5.")

    parser.add_argument("--walk-length",
                        type = int,
                        default = 40,
	                help = "Truncated random walk length. Default is 40.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed for PyTorch. Default is 42.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.025,
	                help = "Learning rate. Default is 0.025.")

    parser.add_argument("--lambd",
                        type = float,
                        default = 0.1,
	                help = "Regularization parameter. Default is 0.1.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 128,
	                help = "Embedding dimensions. Default is 128.")

    parser.add_argument('--workers',
                        type = int,
                        default = 4,
	                help = 'Number of parallel workers. Default is 4.')
    """

    return parser.parse_args()
