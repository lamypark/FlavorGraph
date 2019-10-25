import pandas as pd
import networkx as nx
from tqdm import tqdm, trange
from texttable import Texttable

def graph_reader(input_nodes, input_edges):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """

    print("\nCreating Graphs...")
    graph = nx.Graph()
    graph_ingr_only = nx.Graph()

    print("Nodes Loaded...%s..." % format(input_nodes))
    df_nodes = pd.read_csv(input_nodes)
    for index, row in tqdm(df_nodes.iterrows(), total=len(df_nodes)):
        _, node_id, name, _id, node_type, is_hub = row.values.tolist()
        graph.add_node(node_id, name=name, id=_id, type=node_type, is_hub=is_hub)
        if node_type == 'ingredient':
            graph_ingr_only.add_node(node_id, name=name, id=_id, type=node_type, is_hub=is_hub)

    print("Edges Loaded...%s..." % format(input_edges))
    df_edges = pd.read_csv(input_edges)
    for index, row in tqdm(df_edges.iterrows(), total=len(df_edges)):
        #print(row.values.tolist())
        _, id_1, id_2, score, edge_type = row.values.tolist()
        graph.add_edge(id_1, id_2, weight=score, type=edge_type)
        if edge_type == 'ingr-ingr':
            graph_ingr_only.add_edge(id_1, id_2, weight=score, type=edge_type)

    graph.remove_edges_from(graph.selfloop_edges())
    graph_ingr_only.remove_edges_from(graph.selfloop_edges())

    print("The whole graph - ingredients, food-like compounds, drug-like compounds")
    print("# of nodes in graph: %d" % nx.number_of_nodes(graph))
    print("# of edges in graph: %d" % nx.number_of_edges(graph))

    print("The small graph - ingredients only")
    print("# of nodes in graph: %d" % nx.number_of_nodes(graph_ingr_only))
    print("# of edges in graph: %d" % nx.number_of_edges(graph_ingr_only))

    return graph, graph_ingr_only

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())
