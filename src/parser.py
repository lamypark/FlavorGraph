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
                        default="./input/paths/",
                        type=str, help="input_path_metapaths")
    parser.add_argument('--input_path_deepwalkpaths',
                        default="./input/paths/",
                        type=str, help="input_path_metapaths")

    parser.add_argument('--output_path',
                        default="./output/",
                        type=str, help="output_path")

    # Skip-Gram
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--window_size', default=3, type=int, help="context window size")
    parser.add_argument('--iterations', default=30, type=int, help="iterations")
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    parser.add_argument('--care_type', default=0, type=int, help="if 1, heterogeneous negative sampling, else normal negative sampling")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--min_count', default=5, type=int, help="min count")
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")

    # Graph2vec - common
    parser.add_argument('--skip_paths', default=False, action="store_true")

    # Node2vec - DeepWalker
    parser.add_argument('--num_walks_deepwalk', default=50, type=int, help="number of walks")
    parser.add_argument('--which_deepwalk', default='weighted_small_walk', type=str)
    parser.add_argument('--len_deepwalk', default=5, type=int, help="length of deepwalk")

    # Metapath2vec - MetapathWalker
    parser.add_argument('--num_walks_metapath', default=50, type=int, help="number of walks")
    parser.add_argument('--which_metapath', default='starthub', type=str)
    parser.add_argument('--num_metapath', default=10, type=int, help="number of metapath")
    parser.add_argument('--len_metapath', default=20, type=int, help="length of metapath")
    parser.add_argument('--use_pretrained', default=False, type=int, help="store_true")

    # Pytorch
    parser.add_argument("--seed",
                        type = int,
                        default = 42, help = "Random seed for PyTorch. Default is 42.")

    return parser.parse_args()
