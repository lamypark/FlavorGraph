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
    print("\nPlot Embedding...")
    node2node_name={}
    node_name2is_hub={}
    for node in graph.nodes():
        node_info = graph.nodes[node]
        node_name = node_info['name']
        node2node_name[node] = node_name
        node_name2is_hub[node_name] = node_info['is_hub']

    file = "{}{}-embedding_{}-metapath_{}-dim_{}-initial_lr_{}-window_size_{}-iterations_{}-min_count-_{}-isCSP_{}-CSPcoef.pickle".format(
                        args.output_path, args.idx_embed, args.idx_metapath, args.dim, args.initial_lr, args.window_size, args.iterations, args.min_count, args.CSP_train, args.CSP_coef)

    with open(file, "rb") as pickle_file:
        vectors = pickle.load(pickle_file)
    node_name2vec = {}
    for node in vectors:
        node_name = node2node_name[int(node)]
        node_name2vec[node_name] = vectors[node]
        
    # TSNE    
    node_name2vec_tsne = load_TSNE(node_name2vec, dim=2)
        
    # SAVE
    save_path = file
    plot_category(node_name2vec, node_name2vec_tsne, save_path, node2node_name, node_name2is_hub, True)

    # For Binary Vectors
    if args.CSP_train:
        file = file.replace('.pickle', '_CSPLayer.pickle')
        with open(file, "rb") as pickle_file:
            vectors = pickle.load(pickle_file)
        node_name2vec = {}
        for node in vectors:
            node_name = node2node_name[int(node)]
            node_name2vec[node_name] = vectors[node]
            
        # TSNE    
        node_name2vec_tsne = load_TSNE(node_name2vec, dim=2)
        
        # SAVE
        save_path = file
        plot_category(node_name2vec, node_name2vec_tsne, save_path, node2node_name, node_name2is_hub, True)
    return

def plot_category(node_name2vec, node_name2vec_tsne, path, node2node_name, node_name2is_hub, withLegends=False):
    #Label Load
    
    #print(node2is_hub)
    
    labels = []
    for label in node_name2vec:
        labels.append(label)

    #Legend Load
    if withLegends:
        categories = []
        for label in labels:
            try:
                if node_name2is_hub[label] == 'hub':
                    categories.append('Hub_Ingredient')
                elif node_name2is_hub[label] == 'no_hub':
                    categories.append('Non_hub_Ingredient')
                elif node_name2is_hub[label] == 'food':
                    categories.append('Food_like_Compound')
                elif node_name2is_hub[label] == 'drug':
                    categories.append('Drug_like_Compound')
                else:
                    print(label)
            except KeyError:
                print(label)
                categories.append("None")
        categories_color = list(set(categories))

        category2color = {
        'Hub_Ingredient' :  sns.xkcd_rgb["orange"],
        'Non_hub_Ingredient' : sns.xkcd_rgb["red"],
        'Food_like_Compound' : sns.xkcd_rgb["blue"],
        'Drug_like_Compound'  : sns.xkcd_rgb["green"],
        'None'  : sns.xkcd_rgb["black"]
        }
        
        category2marker = {
        'Hub_Ingredient' : 'diamond-x',
        'Non_hub_Ingredient' : 'square',
        'Food_like_Compound' : 'circle',
        'Drug_like_Compound' : 'circle'
        }
        
        category2size = {
        'Hub_Ingredient' : 14,
        'Non_hub_Ingredient' : 8,
        'Food_like_Compound' : 8,
        'Drug_like_Compound' : 9
        }
        
        label2plot = {
        'Hub_Ingredient' : 50,
        'Non_hub_Ingredient' : 100,
        'Food_like_Compound' : 100,
        'Drug_like_Compound' : 50
        }
        
        category_order = ['Non_hub_Ingredient', 'Food_like_Compound', 'Drug_like_Compound', 'Hub_Ingredient']

        make_plot_with_labels_legends(name=path,
        points=node_name2vec_tsne,
        labels=labels,
        label_to_plot=label2plot,
        legend_labels=categories,
        legend_order=category_order,
        legend_label_marker=category2marker,
        legend_label_size=category2size,
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
def make_plot_with_labels_legends(name, points, labels, label_to_plot, legend_labels, legend_order, legend_label_to_color, legend_label_marker, legend_label_size, pretty_legend_label, publish):
    lst = zip(points, labels, legend_labels)
    full = sorted(lst, key=lambda x: x[2])
    traces = []
    annotations = []
    
    for legend_label, group in itertools.groupby(full, lambda x: x[2]):
        group_points = []
        group_labels = []
        for tup in group:
            point, label, _ = tup
            group_points.append(point)
            group_labels.append(label)
            
        # label, legend_label
        # markers
        group_points = np.stack(group_points)
        traces.append(go.Scattergl(
            x = group_points[:, 0],
            y = group_points[:, 1],
            mode = 'markers',
            marker = dict(
                symbol = legend_label_marker[legend_label],
                color = legend_label_to_color[legend_label],
                size = legend_label_size[legend_label],
                opacity = 0.7,
                line = dict(width = 0.5)
            ),
            text = ['{} ({})'.format(label, pretty_legend_label(legend_label)) for label in group_labels],
            #text = ['{}'.format(label) for label in group_labels],
            #textposition='middle center',
            #textfont=dict(family='sans serif', size = label_to_size[legend_label], color='black'),
            hoverinfo = 'text',
            name = legend_label
        )
        )
        
        res = random.sample(range(1, len(group_points)), label_to_plot[legend_label])
        sampled_group_points = [group_points[i] for i in res]
        sampled_group_labels = [group_labels[i] for i in res]
            
        for point, label in zip(sampled_group_points, sampled_group_labels):
            annotations.append(go.layout.Annotation(
                x=point[0],
                y=point[1],
                xref="x",
                yref="y",
                text=label,
                showarrow=True,
                font=dict(
                    family="Courier New, monospace",
                    size=legend_label_size[legend_label],
                    color="black"
                    ),
                align="center",
                arrowhead=1,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="black",
                ax=0,
                ay=0,
                #bordercolor="#c7c7c7",
                #borderwidth=2,
                #borderpad=4,
                #bgcolor="#ff7f0e",
                opacity=0.8
            )
            )

    # order the legend
    ordered = [[trace for trace in traces if trace.name == lab] for lab in legend_order]
    traces_ordered = flatten(ordered)
    def _set_name(trace):
        trace.name = pretty_legend_label(trace.name)
        return trace
    traces_ordered = list(map(_set_name, traces_ordered))

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
        annotations=annotations,
        title='FlavorNet2.0',   
        font=dict(size=12),
        showlegend=True,
        autosize=True,
        hovermode='closest',
    )
    fig = go.Figure(data=traces_ordered, layout=layout)
    #img_bytes = fig.to_image(format="png")
    #Image(img_bytes)
    if publish:
        plotter = py.iplot
    else:
        plotter = offline.plot
    plotter(fig, filename=name + '.html')
