import random
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import itertools
import datetime
import chart_studio.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from datetime import datetime
from sklearn.manifold import TSNE

# Embedding
def plot_embedding(args, graph, mode=None):
    """
    Plot Embedding
    """
    small_mode = True if 'small' in args.input_nodes else False
    print("\nPlot Embedding...")
    node2ingr={}
    ingr2type={}
    for node in graph.nodes():
        node_info = graph.nodes[node]
        node_name = str(node_info['name'])+"_"+str(node)
        node2ingr[node] = node_name

        if node_info['type'] == 'ingredient':
            ingr2type[node_name] = node_info['type']
        elif node_info['type'] == 'compound':
            ingr2type[node_name] = node_info['is_hub']
        else:
            print("Type Error")

    file = "{}{}-embedding_{}-metapath_{}-dim_{}-initial_lr_{}-window_size_{}-iterations_{}-min_count-_{}-isCSP_{}-CSPcoef_.pickle".format(
                        args.output_path, args.idx_embed, args.idx_metapath, args.dim, args.initial_lr, args.window_size, args.iterations, args.min_count, args.CSP_train, args.CSP_coef)

    with open(file, "rb") as pickle_file:
        vectors = pickle.load(pickle_file)

    print(len(vectors))

    node2vec = {}
    for node in vectors:
        node2vec[int(node)] = vectors[node]
        #print(node)

    # TSNE of node2vec
    node2vec_tsne = load_TSNE(node2vec, dim=2)
    save_path = file.replace("pickle", "html")
    plot_category(node2vec, node2vec_tsne, save_path, node2ingr, ingr2type, True)

    # For Binary Vectors
    if args.CSP_train:
        file = file.replace('.pickle', '_binary.pickle')
        with open(file, "rb") as pickle_file:
            vectors = pickle.load(pickle_file)
            node2vec = {}
        for node in vectors:
            node2vec[int(node)] = vectors[node]
        node2vec_tsne = load_TSNE(node2vec, dim=2)
        save_path = file.replace("pickle", "html")
        plot_category(node2vec, node2vec_tsne, save_path, node2ingr, ingr2type, True)
    return

def plot_category(node2vec, node2vec_tsne, path, node2ingr, ingr2type=None, withLegends=False):
    #Label Load
    labels = []
    for node in node2vec:
        try:
            label = node2ingr[node]
            labels.append(label)
        except KeyError:
            print(node)

    #Legend Load
    if withLegends:
        categories = []
        for label in labels:
            try:
                categories.append(ingr2type[label])
            except KeyError:
                print(label)
                categories.append("None")
        categories_color = list(set(categories))

        category2color = {
        'ingredient' :  sns.xkcd_rgb["red"],
        'food' : sns.xkcd_rgb["blue"],
        'drug' : sns.xkcd_rgb["green"],
        'None'      : sns.xkcd_rgb["black"]
        }

        category_order = ['ingredient', 'food', 'drug']

        make_plot_with_labels_legends(name=path,
        points=node2vec_tsne,
        labels=labels,
        legend_labels=categories,
        legend_order=category_order,
        legend_label_to_color=category2color,
        pretty_legend_label=pretty_category,
        publish=False)

    else:
        make_plot_only_labels(name=path,
                points=node2vec_tsne,
                labels=labels,
                publish=False)

"""
TSNE of Ingredient2Vec

"""
def load_TSNE(node2vec, dim=2):
    print("\nt-SNE Started... ")
    time_start = time.time()

    X = []
    for x in node2vec:
        X.append(node2vec[x])
    tsne = TSNE(n_components=dim)
    X_tsne = tsne.fit_transform(X)

    print("t-SNE done!")
    print("Time elapsed: {} seconds".format(time.time()-time_start))

    return X_tsne


"""
Load functions for plotting a graph
"""

flatten = lambda l: [item for sublist in l for item in sublist]

# Prettify ingredients
pretty_food = lambda s: ' '.join(s.split('_')).capitalize().lstrip()
# Prettify cuisine names
pretty_category = lambda s: ''.join(map(lambda x: x if x.islower() else " "+x, s)).lstrip()

"""
Plot Points with Labels
"""
def make_plot_only_labels(name, points, labels, publish):
    traces = []
    traces.append(go.Scattergl(
            x = points[:, 0],
            y = points[:, 1],
            mode = 'markers',
            marker = dict(
                color = sns.xkcd_rgb["black"],
                size = 8,
                opacity = 0.6,
                #line = dict(width = 1)
            ),
            text = labels,
            hoverinfo = 'text',
        )
        )

    layout = go.Layout(
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            #autotick=True,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            #autotick=True,
            ticks='',
            showticklabels=False
        )
        )

    fig = go.Figure(data=traces, layout=layout)
    if publish:
        plotter = py.iplot
    else:
        plotter = offline.plot
    plotter(fig, filename=name + '.html')

"""
Plot Points with Labels and Legends
"""
def make_plot_with_labels_legends(name, points, labels, legend_labels, legend_order, legend_label_to_color, pretty_legend_label, publish):
    lst = zip(points, labels, legend_labels)
    full = sorted(lst, key=lambda x: x[2])
    traces = []
    for legend_label, group in itertools.groupby(full, lambda x: x[2]):
        group_points = []
        group_labels = []
        for tup in group:
            point, label, _ = tup
            group_points.append(point)
            group_labels.append(label)
        group_points = np.stack(group_points)
        traces.append(go.Scattergl(
            x = group_points[:, 0],
            y = group_points[:, 1],

            mode = 'markers',
            marker = dict(
                color = legend_label_to_color[legend_label],
                size = 8,
                opacity = 0.6,
                #line = dict(width = 1)
            ),
            text = ['{} ({})'.format(label, pretty_legend_label(legend_label)) for label in group_labels],
            hoverinfo = 'text',
            name = legend_label
        )
        )

    # order the legend
    ordered = [[trace for trace in traces if trace.name == lab] for lab in legend_order]
    traces_ordered = flatten(ordered)
    def _set_name(trace):
        trace.name = pretty_legend_label(trace.name)
        return trace
    traces_ordered = list(map(_set_name, traces_ordered))

    """
    annotations = []
    for index in range(50):
        new_dict = dict(
                x=points[:, 0][index],
                y=points[:, 1][index],
                xref='x',
                yref='y',
                text=labels[index],
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-10
            )
        annotations.append(new_dict)
    """

    layout = go.Layout(
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=True,
            showline=True,
            #autotick=True,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=True,
            showline=True,
            #autotick=True,
            ticks='',
            showticklabels=False
        ),
        #annotations=annotations
        title='FlavorNet2.0.html',
        font=dict(size=12),
        showlegend=True,
        autosize=True,
        hovermode='closest',
    )
    fig = go.Figure(data=traces_ordered, layout=layout)
    if publish:
        plotter = py.iplot
    else:
        plotter = offline.plot
    plotter(fig, filename=name + '.html')
