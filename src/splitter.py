import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from walkers import DeepWalker
from ego_splitting import EgoNetSplitter

class Splitter(torch.nn.Module):
     """
     An implementation of "Splitter: Learning Node Representations that Capture Multiple Social Contexts" (WWW 2019).
     Paper: http://epasto.org/papers/www2019splitter.pdf
     """
     def __init__(self, args, base_node_count, node_count):
         """
         Splitter set up.
         :param args: Arguments object.
         :param base_node_count: Number of nodes in the source graph.
         :param node_count: Number of nodes in the persona graph.
         """
         super(Splitter, self).__init__()
         self.args = args
         self.base_node_count = base_node_count
         self.node_count = node_count

     def create_weights(self):
         """
         Creating weights for embedding.
         """
         self.base_node_embedding = torch.nn.Embedding(self.base_node_count, self.args.dimensions, padding_idx = 0)
         self.node_embedding = torch.nn.Embedding(self.node_count, self.args.dimensions, padding_idx = 0)
         self.node_noise_embedding = torch.nn.Embedding(self.node_count, self.args.dimensions, padding_idx = 0)

     def initialize_weights(self, base_node_embedding, mapping):
         """
         Using the base embedding and the persona mapping for initializing the embedding matrices.
         :param base_node_embedding: Node embedding of the source graph.
         :param mapping: Mapping of personas to nodes.
         """
         persona_embedding = np.array([base_node_embedding[original_node] for node, original_node in mapping.items()])
         self.node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
         self.node_noise_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
         self.base_node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(base_node_embedding), requires_grad=False)

     def calculate_main_loss(self, sources, contexts, targets):
         """
         Calculating the main embedding loss.
         :param sources: Source node vector.
         :param contexts: Context node vector.
         :param targets: Binary target vector.
         :return main_loss: Loss value.
         """
         node_f = self.node_embedding(sources)
         node_f = torch.nn.functional.normalize(node_f, p=2, dim=1)
         feature_f = self.node_noise_embedding(contexts)
         feature_f = torch.nn.functional.normalize(feature_f, p=2, dim=1)
         scores = torch.sum(node_f*feature_f,dim=1)
         scores = torch.sigmoid(scores)
         main_loss = targets*torch.log(scores) + (1-targets)*torch.log(1-scores)
         main_loss = -torch.mean(main_loss)
         return main_loss

     def calculate_regularization(self, pure_sources, personas):
         """
         Calculating the regularization loss.
         :param pure_sources: Source nodes in persona graph.
         :param personas: Context node vector.
         :return regularization_loss: Loss value.
         """
         source_f = self.node_embedding(pure_sources)
         original_f = self.base_node_embedding(personas)
         scores = torch.clamp(torch.sum(source_f*original_f,dim=1),-15,15)
         scores = torch.sigmoid(scores)
         regularization_loss = -torch.mean(torch.log(scores))
         return regularization_loss

     def forward(self, sources, contexts, targets, personas, pure_sources):
         """
         Doing a forward pass.
         :param sources: Source node vector.
         :param contexts: Context node vector.
         :param targets: Binary target vector.
         :param pure_sources: Source nodes in persona graph.
         :param personas: Context node vector.
         :return loss: Loss value.
         """
         main_loss = self.calculate_main_loss(sources, contexts, targets)
         regularization_loss = self.calculate_regularization(pure_sources, personas)
         loss = main_loss + self.args.lambd*regularization_loss
         return loss

class SplitterTrainer(object):
    """
    Class for training a Splitter.
    """
    def __init__(self, graph, args, node2ingr):
        """
        :param graph: NetworkX graph object.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args
        self.node2ingr = node2ingr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_noises(self):
        """
        Creating node noise distribution for negative sampling.
        """
        self.downsampled_degrees = {node: int(1+self.egonet_splitter.persona_graph.degree(node)**0.75) for node in self.egonet_splitter.persona_graph.nodes()}
        self.noises = [k for k,v in self.downsampled_degrees.items() for i in range(v)]

    def base_model_fit(self):
        """
        Fitting DeepWalk on base model.
        """
        self.base_walker = DeepWalker(self.graph, self.args)
        print("\nDoing base random walks.\n")
        self.base_walker.create_features()
        print("\nLearning the base model.\n")
        self.base_node_embedding = self.base_walker.learn_base_embedding()
        print("\nDeleting the base walker.\n")
        del self.base_walker

    def create_split(self):
        """
        Creating an EgoNetSplitter.
        """
        self.egonet_splitter = EgoNetSplitter(self.graph)
        self.persona_walker = DeepWalker(self.egonet_splitter.persona_graph, self.args)
        print("\nDoing persona random walks.\n")
        self.persona_walker.create_features()
        self.create_noises()

    def setup_model(self):
        """
        Creating a model and doing a transfer to GPU.
        """
        base_node_count = self.graph.number_of_nodes()
        persona_node_count = self.egonet_splitter.persona_graph.number_of_nodes()
        self.model = Splitter(self.args, base_node_count, persona_node_count)
        self.model.create_weights()
        self.model.initialize_weights(self.base_node_embedding, self.egonet_splitter.personality_map)
        self.model = self.model.to(self.device)

    def transfer_batch(self, source_nodes, context_nodes, targets, persona_nodes, pure_source_nodes):
        """
        Transfering the batch to GPU.
        """
        self.sources = torch.LongTensor(source_nodes).to(self.device)
        self.contexts = torch.LongTensor(context_nodes).to(self.device)
        self.targets = torch.FloatTensor(targets).to(self.device)
        self.personas = torch.LongTensor(persona_nodes).to(self.device)
        self.pure_sources = torch.LongTensor(pure_source_nodes).to(self.device)

    def optimize(self):
        """
        Doing a weight update.
        """
        loss = self.model(self.sources, self.contexts, self.targets, self.personas, self.pure_sources)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def process_walk(self, walk):
        """
        Process random walk (source, context) pairs. Sample negative instances and create persona node list.
        :param walk: Random walk sequence.
        """
        left_nodes = [walk[i] for i in range(len(walk)-self.args.window_size) for j in range(1, self.args.window_size+1)]
        right_nodes = [walk[i+j] for i in range(len(walk)-self.args.window_size) for j in range(1, self.args.window_size+1)]
        node_pair_count = len(left_nodes)
        source_nodes = left_nodes + right_nodes
        context_nodes = right_nodes + left_nodes
        persona_nodes = np.array([self.egonet_splitter.personality_map[source_node] for source_node in source_nodes])
        pure_source_nodes = np.array(source_nodes)
        source_nodes = np.array((self.args.negative_samples+1)*source_nodes)
        context_nodes = np.concatenate((np.array(context_nodes), np.random.choice(self.noises,node_pair_count*2*self.args.negative_samples)))
        positives = [1.0 for node in range(node_pair_count*2)]
        negatives = [0.0 for node in range(node_pair_count*self.args.negative_samples*2)]
        targets = np.array(positives + negatives)
        self.transfer_batch(source_nodes, context_nodes, targets, persona_nodes, pure_source_nodes)

    def update_average_loss(self, loss_score):
        """
        Updating the average loss and the description of the time remains bar.
        :param loss_score: Loss on the sample.
        """
        self.cummulative_loss = self.cummulative_loss + loss_score
        self.steps = self.steps + 1
        average_loss = self.cummulative_loss/self.steps
        self.walk_steps.set_description("Splitter (Loss=%g)" % round(average_loss,4))

    def reset_average_loss(self, step):
        """
        Doing a reset on the average loss.
        :param step: Current number of walks processed.
        """
        if step % 100 == 0:
            self.cummulative_loss = 0
            self.steps = 0

    def fit(self):
        """
        Fitting a model.
        """
        self.base_model_fit()
        self.create_split()
        self.setup_model()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        print("\nLearning the joint model.\n")
        random.shuffle(self.persona_walker.paths)
        self.walk_steps = trange(len(self.persona_walker.paths), desc="Loss")
        for step in self.walk_steps:
            self.reset_average_loss(step)
            walk = self.persona_walker.paths[step]
            self.process_walk(walk)
            loss_score = self.optimize()
            self.update_average_loss(loss_score)

    def walk(self):
        """
        Fitting DeepWalk on base model.
        self.base_model_fit()
        """
        print("\nDoing base random walks for a whole graph.\n")
        self.base_walker = DeepWalker(self.graph, self.args)

        # random walk를 recipe walk로 대체 가능!
        self.base_walker.create_features()

        # Printing walks
        for path in self.base_walker.paths:
            start_node = path[0]
            walk_nodes = path[1:]
            #print(self.node2ingr[start_node])
            #print([self.node2ingr[node] for node in walk_nodes])
        #print()

    def split(self):
        """
        Creating an EgoNetSplitter.
        self.create_split()
        """
        print("\nEgoNet Split.\n")
        self.egonet_splitter = EgoNetSplitter(self.graph)

        for node in self.graph.nodes()[:10]:
            personalities = self.egonet_splitter.personalities[node]
            persona2node = self.egonet_splitter.personality_map
            components = self.egonet_splitter.components[node]

            print(components)

            persona2components = {}
            for other_node, persona in components:
                if persona in persona2components:
                    persona2components[persona] += [other_node]
                else:
                    persona2components[persona] = [other_node]

            # if len(personalities) > 1:
            #     print(node, self.node2ingr[node])
            #     print(personalities)
            #     print(persona2components)
            #
            #     for persona in persona2components:
            #         node = persona2node[persona]
            #         components = persona2components[persona]
            #         ingrs = []
            #         for component in components:
            #             node = persona2node[component]
            #             ingr = self.node2ingr[node]
            #             ingrs += [ingr]
            #         print(ingrs)
            #     print()


        # self.create_egonets()
        # self.map_personalities()
        # self.create_persona_graph()
        #print(len(self.egonet_splitter.components))
        #print(len(self.egonet_splitter.personalities))
        #print(len(self.egonet_splitter.persona_graph))
        #for persona_graph_edge in self.egonet_splitter.persona_graph_edges:
        #    print(persona_graph_edge)

        # self.persona_walker = DeepWalker(self.egonet_splitter.persona_graph, self.args)
        # print("\nDoing persona random walks.\n")
        # self.persona_walker.create_features()
        # self.create_noises()

    def save_embedding(self):
        """
        Saving the node embedding.
        """
        print("\n\nSaving the model.\n")
        nodes = [node for node in self.egonet_splitter.persona_graph.nodes()]
        nodes.sort()
        nodes = torch.LongTensor(nodes).to(self.device)
        self.embedding = self.model.node_embedding(nodes).cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        self.embedding  = np.concatenate([np.array(range(self.embedding.shape[0])).reshape(-1,1),self.embedding],axis=1)
        self.embedding = pd.DataFrame(self.embedding, columns = embedding_header)
        self.embedding.to_csv(self.args.embedding_output_path, index = None)

    def save_persona_graph_mapping(self):
        """
        Saving the persona map.
        """
        with open(self.args.persona_output_path, "w") as f:
           json.dump(self.egonet_splitter.personality_map, f)
