import pandas as pd
import pickle
import networkx as nx
from tqdm import tqdm, trange
from texttable import Texttable

def graph_reader(input_nodes, input_edges):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    print("\n\n##########################################################################")
    print("### Creating Graphs...")
    graph = nx.Graph()
    graph_ingr_only = nx.Graph()

    print("Nodes Loaded...%s..." % format(input_nodes))
    df_nodes = pd.read_csv(input_nodes)
    for index, row in tqdm(df_nodes.iterrows(), total=len(df_nodes)):
        node_id, name, _id, node_type, is_hub = row.values.tolist()
        graph.add_node(node_id, name=name, id=_id, type=node_type, is_hub=is_hub)
        if node_type == 'ingredient':
            graph_ingr_only.add_node(node_id, name=name, id=_id, type=node_type, is_hub=is_hub)

    print("Edges Loaded...%s..." % format(input_edges))
    df_edges = pd.read_csv(input_edges)
    for index, row in tqdm(df_edges.iterrows(), total=len(df_edges)):
        #print(row.values.tolist())
        id_1, id_2, score, edge_type = row.values.tolist()
        graph.add_edge(id_1, id_2, weight=score, type=edge_type)
        if edge_type == 'ingr-ingr':
            graph_ingr_only.add_edge(id_1, id_2, weight=score, type=edge_type)

    #graph.remove_edges_from(graph.selfloop_edges())
    #graph_ingr_only.remove_edges_from(graph.selfloop_edges())

    print("\nThe whole graph - ingredients, food-like compounds, drug-like compounds")
    print("# of nodes in graph: %d" % nx.number_of_nodes(graph))
    print("# of edges in graph: %d" % nx.number_of_edges(graph))

    #print("The small graph - ingredients only")
    #print("# of nodes in graph: %d" % nx.number_of_nodes(graph_ingr_only))
    #print("# of edges in graph: %d" % nx.number_of_edges(graph_ingr_only))

    return graph, graph_ingr_only

def evaluate(args, graph):
    """
    Downstream Applications
    Evaluation
    """
    print("\nEvaluation...")

    node2node_name={}
    node_name2node={}

    for node in graph.nodes():
        node_info = graph.nodes[node]
        node_name = node_info['name']
        node2node_name[node] = node_name
        node_name2node[node_name] = node

    csv = "./input/node_classification_hub.csv"
    df = pd.read_csv(csv)
    categories = df.columns

    file = "{}{}-embedding_{}-metapath_{}-dim_{}-initial_lr_{}-window_size_{}-iterations_{}-min_count-_{}-isCSP_{}-CSPcoef.pickle".format(
                        args.output_path, args.idx_embed, args.idx_metapath, args.dim, args.initial_lr, args.window_size, args.iterations, args.min_count, args.CSP_train, args.CSP_coef)

    with open(file, "rb") as pickle_file:
        vectors = pickle.load(pickle_file)

    node_name2vec={}
    for node in vectors:
        node_name = node2node_name[int(node)]
        node_name2vec[node_name] = vectors[node]

    X=[]
    y=[]
    for category in categories:
        ingredients = df[category].values
        for name in ingredients:
            vec = node_name2vec[name]
            X.append(vec)
            y.append(category)


    train_ratios = [0.2, 0.4, 0.6, 0.8]
    for ratio in train_ratios:
        print("\nratio:", ratio)
        train(X, y, ratio)

    # For Binary Vectors
    if args.CSP_train:
        file = file.replace('.pickle', '_CSPLayer.pickle')
        with open(file, "rb") as pickle_file:
            vectors = pickle.load(pickle_file)
        node_name2vec={}
        for node in vectors:
            node_name = node2node_name[int(node)]
            node_name2vec[node_name] = vectors[node]
        X=[]
        y=[]
        for category in categories:
            ingredients = df[category].values
            for name in ingredients:
                vec = node_name2vec[name]
                X.append(vec)
                y.append(category)

        train_ratios = [0.2, 0.4, 0.6, 0.8]
        for ratio in train_ratios:
            print("\nratio:", ratio)
            train(X, y, ratio)

    return

def train(X, y, train_ratio):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    test_ratio = 1-train_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=36)

    """
    Classification
    """
    #clf = LogisticRegression(C=1000.0, random_state=0).fit(X_train, y_train)
    clf = svm.SVC(kernel='linear', C=1e20).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    #print(y_test)
    #print(y_pred)
    #print("accuracy: %.2f" %accuracy_score(y_test, y_pred))
    #print("Precision : %.3f" % precision_score(y_test, y_pred))
    #print("Recall : %.3f" % recall_score(y_test, y_pred))
    #print("F1-micro : %.3f" % f1_score(y_test, y_pred, average='micro'))
    #print("F1-macro : %.3f" % f1_score(y_test, y_pred, average='macro'))
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print("F1-macro")
    print(f1_macro)
    print("F1-micro")
    print(f1_micro)

    """
    Clustering
    """

    from sklearn.metrics.cluster import normalized_mutual_info_score
    from nltk.cluster import KMeansClusterer
    import nltk

    NUM_CLUSTERS=8
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    #set(assigned_clusters)
    nmi = normalized_mutual_info_score(assigned_clusters, y)
    print("NMI")
    print(nmi)

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
