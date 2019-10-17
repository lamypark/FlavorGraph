import networkx as nx
from tqdm import tqdm

class EgoNetSplitter(object):
    """
    A lightweight implementation of "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters".
    Paper: https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf
    Video: https://www.youtube.com/watch?v=xMGZo-F_jss
    Slides: https://epasto.org/papers/kdd2017-Slides.pdf
    """
    def __init__(self, graph):
        """
        :param graph: Networkx object.
        :param resolution: Resolution parameter of Python Louvain.
        """
        self.graph = graph
        self.create_egonets()
        self.map_personalities()
        self.create_persona_graph()

    def create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.
        :param node: Node ID for egonet (ego node).
        """
        ego_net_minus_ego = self.graph.subgraph(self.graph.neighbors(node))
        components = {i: nodes for i, nodes in enumerate(nx.connected_components(ego_net_minus_ego))}

        print(node)
        print(ego_net_minus_ego)
        print(nx.connected_components(ego_net_minus_ego))
        print(components)


        new_mapping = {}
        personalities = []
        for k, v in components.items():
            personalities.append(self.index)
            for other_node in v:
                new_mapping[other_node] = self.index
            self.index = self.index+1
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def create_egonets(self):
        """
        Creating an egonet for each node.
        """
        self.components = {}
        self.personalities = {}
        self.index = 0
        print("\nCreating egonets.\n")
        #for node in tqdm(self.graph.nodes()):
        for node in range(10):
            self.create_egonet(node)

    def map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {persona: node for node in self.graph.nodes() for persona in self.personalities[node]}

    def create_persona_graph(self):
        """
        Create a persona graph using the egonet components.
        """
        print("\nCreating the persona graph.\n")
        self.persona_graph_edges = [(self.components[edge[0]][edge[1]], self.components[edge[1]][edge[0]]) for edge in tqdm(self.graph.edges())]
        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)
