import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

from dataloader import DataReader, DatasetLoader
from walkers import DeepWalker, MetaPathWalker
from model import SkipGramModel, SkipGramModelAug

class Metapath2Vec:
    def __init__(self, args, graph):
        # 1. generate walker
        walker = MetaPathWalker(args, graph)
        self.metapaths = walker.generate_metapaths(args)
        if not args.skip_paths:
            walker.create_metapath_walks(args, args.num_walks_metapath, self.metapaths)
        self.inputFileName = "{}metapath_{}-whichmeta_{}-num_walks_{}-len_walk_{}-num_metapath_{}-dim.txt".format(args.input_path_metapaths, args.which_metapath, args.num_walks_metapath, args.len_metapath, args.num_metapath, args.dim)

        # 2. read data
        self.data = DataReader(args.min_count, args.care_type, self.inputFileName)

        # 3. make dataset for training
        dataset = DatasetLoader(self.data, args.window_size)

        # 4. initialize dataloader
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        # 5. load pre-trained node2vec for ingredients
        # if not args.use_pretrained:
        #     print("use_pretrained")
        #     self.pretrained_file_name = "{}embedding_deepwalk_{}-whichmeta_{}-num_walks_{}-len_walk_{}-dim.pickle".format(args.output_path, args.which_deepwalk, args.num_walks_deepwalk, args.len_deepwalk, args.dim)
        #     with open(self.pretrained_file_name, "rb") as pickle_file:
        #         vectors = pickle.load(pickle_file)
        #
        #     self.pretrained_weights = []
        #     for wid, w in self.data.id2word.items():
        #         if w in vectors:
        #             vector = np.array(vectors[w])
        #         else:
        #             vector = np.array(torch.randn(args.dim))
        #         self.pretrained_weights.append(vector)
        #
        #     self.pretrained_weights = np.array(self.pretrained_weights)
        #     #print(self.pretrained_weights)"

        self.output_file_name = "{}embedding_metapath_{}-whichmeta_{}-num_walks_{}-len_walk_{}-num_metapath_{}-dim.pickle".format(args.output_path, args.which_metapath, args.num_walks_metapath, args.len_metapath, args.num_metapath, args.dim)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr

        self.skip_gram_model = SkipGramModelAug(self.emb_size, self.emb_dimension, pretrained_weights=None, is_metapath=False, nodes=self.data.id2word)
        # self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, pretrained_weights=None, is_metapath=False)

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

        self.inputFileName = "{}deepwalk_{}-whichmeta_{}-num_walks_{}-len_walk_{}-dim.txt".format(args.input_path_deepwalkpaths, args.which_deepwalk, args.num_walks_deepwalk, args.len_deepwalk, args.dim)

        # 2. read data
        self.data = DataReader(args.min_count, args.care_type, self.inputFileName)

        # 3. make dataset for training
        dataset = DatasetLoader(self.data, args.window_size)

        # 4. initialize dataloader
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        self.output_file_name = "{}embedding_deepwalk_{}-whichmeta_{}-num_walks_{}-len_walk_{}-dim.pickle".format(args.output_path, args.which_deepwalk, args.num_walks_deepwalk, args.len_deepwalk, args.dim)
        #self.output_file_name = self.inputFileName.replace("deepwalk", "embedding_deepwalk").replace("txt", "pickle")
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, pretrained_weights=None, is_metapath=False)

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
                    #if i > 0 and i % 300 == 0:
            print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
