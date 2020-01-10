import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
from dataloader import DataReader, DatasetLoader
from walkers import MetaPathWalker
from model import SkipGramModel, SkipGramModelAux


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Metapath2Vec:
    def __init__(self, args, graph):
        # 1. generate walker
        walker = MetaPathWalker(args, graph)

        files = os.listdir(args.input_path_metapaths)
        is_file = False
        for file in files:
            fullFilename = os.path.join(args.input_path_metapaths, file)
            # if file exists, load the file.
            if file.startswith(args.idx_metapath):
                is_file = True
                print("\n !!! Found the file that you have specified...")
                self.inputFileName = "{}{}-metapath_{}-whichmeta_{}-num_walks_{}-len_metapath.txt".format(args.input_path_metapaths, args.idx_metapath, args.which_metapath, args.num_walks, args.len_metapath)
                print("### Metapaths Loaded...", self.inputFileName)

        # if file does not exists, create the new one.
        if not is_file:
            print("\n !!! There is no metapaths with the given parameters...")
            print("### Creating new Metapaths...")
            self.metapaths = walker.generate_metapaths(args)
            walker.create_metapath_walks(args, args.num_walks, self.metapaths)
            self.inputFileName = "{}{}-metapath_{}-whichmeta_{}-num_walks_{}-len_metapath.txt".format(args.input_path_metapaths, args.idx_metapath, args.which_metapath, args.num_walks, args.len_metapath)
            print("### Metapaths Loaded...", self.inputFileName)

        # 2. read data
        print("\n\n##########################################################################")
        print("### Metapaths to DataLoader...", self.inputFileName)
        self.data = DataReader(args.min_count, args.care_type, self.inputFileName)

        # 3. make dataset for training
        dataset = DatasetLoader(self.data, args.window_size)


        # 4. initialize dataloader
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)
        self.output_file_name = "{}{}-embedding_{}-metapath_{}-dim_{}-initial_lr_{}-window_size_{}-iterations_{}-min_count-_{}-isCSP_{}-CSPcoef.pickle".format(
                            args.output_path, args.idx_embed, args.idx_metapath, args.dim, args.initial_lr, args.window_size, args.iterations, args.min_count, args.CSP_train, args.CSP_coef)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.aux_mode = args.CSP_train
        self.aux_coef = args.CSP_coef

        if args.CSP_train:
            print("\n\n#####################################")
            print("### SkipGram with CSP")
            self.skip_gram_model = SkipGramModelAux(self.emb_size, self.emb_dimension, nodes=self.data.id2word, aux_coef=self.aux_coef, CSP_save=args.CSP_save)
        else:
            print("\n\n#####################################")
            print("### SkipGram Normal")
            self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        for iteration in range(self.iterations):
            print(self.skip_gram_model.u_embeddings.weight.data)

            print("\n\n\nIteration: " + str(iteration + 1))
            # Temporary Fix!
            if self.aux_mode:
                u = self.skip_gram_model.u_embeddings.weight
                v = self.skip_gram_model.v_embeddings.weight
                e = self.skip_gram_model.encoder.weight
                optimizer = optim.Adam([u, v], lr=self.initial_lr)
                aux_optimizer = optim.Adam([e], lr=0.003)
                aux_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(aux_optimizer, len(self.dataloader))
            else:
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
                    if self.aux_mode:
                        aux_scheduler.step()
                        aux_optimizer.zero_grad()

                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    if self.aux_mode:
                        aux_optimizer.step()
                    running_loss = running_loss * 0.9 + loss.item() * 0.1

                    if i > 0 and i % int(len(self.dataloader)/3) == 0:
                        print(" Loss: " + str(running_loss))
                        if self.aux_mode:
                            print(" Auxiliary Loss: " + str(self.skip_gram_model.aux_loss.item()))

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
