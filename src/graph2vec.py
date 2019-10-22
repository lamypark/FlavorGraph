import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataloader import DataReader, DatasetLoader
from walkers import DeepWalker, MetaPathWalker
from model import SkipGramModel

class Metapath2Vec:
    def __init__(self, args, graph):
        # 1. generate walker
        walker = MetaPathWalker(args, graph)
        self.metapaths = walker.generate_metapaths(args)
        if not args.skip_paths:
            walker.create_metapath_walks(args, args.num_walks, self.metapaths)
        self.inputFileName = "{}metapaths_{}-meta_{}-nodes_{}-paths_{}-walks_{}-dim.txt".format(args.input_path_metapaths, args.which_metapath, args.len_metapaths, args.num_metapaths, args.num_walks, args.dim)

        # 2. read data
        self.data = DataReader(args.min_count, args.care_type, args.num_walks, self.inputFileName)

        # 3. make dataset for training
        dataset = DatasetLoader(self.data, args.window_size)

        # 4. initialize dataloader
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        self.output_file_name = "{}embedding_metapath_{}-meta_{}-nodes_{}-paths_{}-walks_{}-dim.pickle".format(args.output_path, args.which_metapath, args.len_metapaths, args.num_metapaths, args.num_walks, args.dim)
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
                    if i > 0 and i % 300 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)

class Node2Vec:
    def __init__(self, args, graph):
        print("\nPerforming Node2vec...\n")
        # 1. generate walker
        walker = DeepWalker(args, graph)
        print("\nDoing deepwalks...\n")
        if not args.skip_paths:
            walker.create_deepwalk_paths()

        self.inputFileName = "{}deepwalkpaths_{}-whichmeta_{}-nodes_{}-paths_{}-walks_{}-dim.txt".format(args.input_path_deepwalkpaths, args.which_deepwalkpath, args.len_deepwalkpaths, args.num_deepwalkpaths, args.num_walks, args.dim)

        # 2. read data
        self.data = DataReader(args.min_count, args.care_type, args.num_walks, self.inputFileName)

        # 3. make dataset for training
        dataset = DatasetLoader(self.data, args.window_size)

        # 4. initialize dataloader
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        self.output_file_name = "{}embedding_deepwalkpath_{}-whichmeta_{}-nodes_{}-paths_{}-walks_{}-dim.pickle".format(args.output_path, args.which_deepwalkpath, args.len_deepwalkpaths, args.num_deepwalkpaths, args.num_walks, args.dim)
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
                    if i > 0 and i % 300 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
