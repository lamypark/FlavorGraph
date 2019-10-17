import pandas as pd
import networkx as nx
from texttable import Texttable

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

def graph_reader(node2ingr, edge_id2score):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.Graph()
    nodes = sorted(list(node2ingr.keys()))
    graph.add_nodes_from(nodes)
    for edge in edge_id2score:
        graph.add_edge(edge[0], edge[1], weight=edge_id2score[edge])
    graph.remove_edges_from(graph.selfloop_edges())

    for node in nodes[:10]:
        ego_net_minus_ego = graph.subgraph(graph.neighbors(node))
        if len(list(nx.connected_components(ego_net_minus_ego))) > 1:
            print(node, node2ingr[node])

            print(list(graph.neighbors(node)))
            print(ego_net_minus_ego.edges)

            for i in nx.connected_components(ego_net_minus_ego):
                print(i)

        print()



    print("# nodes in graph: %d" % nx.number_of_nodes(graph))
    print("# edges in graph: %d" % nx.number_of_edges(graph))

    #print(nx.get_edge_attributes(graph, 'weight')[(101, 214)])

    return graph
