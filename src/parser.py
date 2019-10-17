import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the Chameleons dataset.
    The default hyperparameters give a good quality representation without grid search.
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

    return parser.parse_args()
