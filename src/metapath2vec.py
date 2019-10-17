import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from reading_data import DataReader, Metapath2vecDataset
from model import SkipGramModel

import numpy as np

def generate_metapaths(args):
    if args.make_meta == 'random':
        return [list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], args.len_metapaths, p=[0.3, 0.25, 0.45])) for _ in range(args.num_metapaths)]
    elif args.make_meta == 'starthub':
        return_list = []
        for _ in range(args.num_metapaths):
            small_list = ['hub+ingredient']
            remain_list = list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], args.len_metapaths-1, p=[0.3, 0.25, 0.45]))
            return_list.append(small_list + remain_list)
        return return_list
    elif args.make_meta == 'hxxhxxh':
        return_list = []
        for _ in range(args.num_metapaths):
            small_list = ['hub+ingredient']
            remain_list1 = list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], int(args.len_metapaths/2), p=[0.4, 0.4, 0.2]))
            remain_list2 = list(np.random.choice(['no_hub+ingredient', 'food_comp', 'hub+ingredient'], int(args.len_metapaths/2), p=[0.4, 0.4, 0.2]))
            return_list.append(small_list + remain_list1 + small_list + remain_list2 + small_list)
        return return_list
    else:
        return None

class Metapath2VecTrainer:
    def __init__(self, args):
        metapaths = generate_metapaths(args)
        self.data = DataReader(args.download, args.min_count, args.care_type, args.num_walks, metapaths, args.skip_meta)
        dataset = Metapath2vecDataset(self.data, args.window_size)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        self.output_file_name = "./output/flavornet_embeddings_{}-meta_{}-nodes_{}-paths_{}-walks_{}-dim.pickle".format(args.make_meta, args.len_metapaths, args.num_metapaths, args.num_walks, args.dim)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Metapath2vec")
    #parser.add_argument('--input_file', type=str, help="input_file")
    parser.add_argument('--download', type=str, help="download_path")
    parser.add_argument('--output_file', type=str, help='output_file')
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--window_size', default=7, type=int, help="context window size")
    parser.add_argument('--iterations', default=5, type=int, help="iterations")
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    parser.add_argument('--care_type', default=0, type=int, help="if 1, heterogeneous negative sampling, else normal negative sampling")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--min_count', default=5, type=int, help="min count")
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")

    parser.add_argument('--num_walks', default=10, type=int, help="number of walks")
    parser.add_argument('--num_metapaths', default=10, type=int, help="number of metapaths")
    parser.add_argument('--len_metapaths', default=10, type=int, help="length of metapaths")
    parser.add_argument('--make_meta', default='random', type=str)
    parser.add_argument('--skip_meta', default=False, action="store_true")


    args = parser.parse_args()
    m2v = Metapath2VecTrainer(args)
    m2v.train()
