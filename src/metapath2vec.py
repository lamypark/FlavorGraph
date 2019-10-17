import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from walkers import MetaPathWalker
from model import SkipGramModel

class Metapath2Vec:
    def __init__(self, args, graph):

        # 1. generate metapaths
        metapathwalker = MetaPathWalker(args, graph)
        self.metapaths = metapathwalker.generate_metapaths(args)
        if not args.skip_meta:
            metapathwalker.create_metapath_walks(args.num_walks, self.metapaths)
        self.inputFileName = "./input/flavornet_metapaths_{}_{}.txt".format(args.num_walks, len(self.meta_paths))

        # 2. read data
        #self.data = DataReader(args.min_count, args.care_type, args.num_walks, self.metapaths, self.inputFileName, args.skip_meta)

        """
        # 3. make dataset for training
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

        """

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

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8
    def __init__(self, min_count, care_type, num_walks, meta_paths, inputFileName, skip_meta=False):
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = inputFileName

        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="ISO-8859-1"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Metapath2vecDataset(Dataset):
    def __init__(self, data, window_size):
        # read in data, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="ISO-8859-1")

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            pair_catch.append((u, v, self.data.getNegatives(v,5)))
                    return pair_catch


    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
