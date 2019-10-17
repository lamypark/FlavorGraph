import random

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from datetime import datetime

def plot_graph(graph):
    df_cate = pd.read_csv("./input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv", sep =",")
    ingr2cate = df_cate.set_index('ingredient')['category'].to_dict()

    node2cate = {}
    inodes = []
    fnodes = []
    dnodes = []

    for node in graph.nodes():
        node_info = graph.nodes[node]
        if node_info['type'] == 'ingredient':
            ingr = node_info['name']
            node2cate[node] = ingr2cate[ingr]
            inodes.append(node)
        else:
            fcnodes.append(node)
    nx.set_node_attributes(graph,node2cate,name='category')

    ipos = custom_cate_layout(graph)
    fpos = circular_layout(fnodes)
    dpos = circular_layout(dnodes)

    return

def make_trace_nodes(node_id2pos, labels, label_sizes, legend_labels, legend_order, legend_label_to_color):
    lst = zip(node_id2pos, labels, legend_labels, label_sizes)
    full = sorted(lst, key=lambda x: x[2])
    traces = []
    for legend_label, group in itertools.groupby(full, lambda x: x[2]):
        group_points = []
        group_sizes = []
        group_labels = []
        for tup in group:
            node_id, label, _, size = tup
            point = node_id2pos[node_id]
            group_points.append(point)
            group_sizes.append(size)
            group_labels.append(label)
        #print([x/3 for x in group_sizes])
        group_points = np.stack(group_points)
        #print(group_points)
        traces.append(go.Scatter(
            x = group_points[:, 0],
            y = group_points[:, 1],
            mode = 'markers+text',
            marker = dict(
                color = legend_label_to_color[legend_label],
                size = group_sizes,
                opacity = 0.6,
                #line = dict(width = 1)
            ),
            text = group_labels,
            textposition='middle center',
            textfont=dict(family='sans serif', size=[x/3 if x > 8 else 0 for x in group_sizes], color='black'),
            hoverinfo = 'none',
            name = legend_label
        )
        )
    #print(traces)
    # order the legend
    ordered = [[trace for trace in traces if trace.name == lab] for lab in legend_order]
    #print(ordered)
    flatten = lambda l: [item for sublist in l for item in sublist]
    traces_ordered = flatten(ordered)
    #print(traces_ordered)

    return traces_ordered

def draw():
    """
    Making Node Traces
    """
    ###############################################
    # co-ordinates of nodes, vertecies
    ###############################################
    node_sizes = []
    categories = []

    category2color = {
        'Bakery/Dessert/Snack' :  sns.xkcd_rgb["brown"],
        'Beverage' : sns.xkcd_rgb["forest green"],
        'Beverage Alcoholic' : sns.xkcd_rgb["wine"],
        'Cereal/Crop/Bean' : sns.xkcd_rgb["mustard yellow"],
        'Dairy' : sns.xkcd_rgb["yellow"],
        'Dish/End Product' : sns.xkcd_rgb["purple"],
        'Essential Oil/Fat' : sns.xkcd_rgb["cream"],
        'Flower' : sns.xkcd_rgb["blue"],
        'Fruit' : sns.xkcd_rgb["lime"],
        'Fungus' : sns.xkcd_rgb["mushroom"],
        'Nut/Seed' : sns.xkcd_rgb["khaki"],
        'Meat/Animal Product' : sns.xkcd_rgb["red"],
        'Plant/Vegetable' : sns.xkcd_rgb["green"],
        'Sauce/Powder/Dressing' : sns.xkcd_rgb["plum purple"],
        'Seafood' : sns.xkcd_rgb["cyan"],
        'Spice' : sns.xkcd_rgb["pink"],
        'ETC' : sns.xkcd_rgb["pale blue"]
    }

    category_order = [
        'Bakery/Dessert/Snack',
        'Beverage',
        'Beverage Alcoholic',
        'Cereal/Crop/Bean',
        'Dairy',
        'Dish/End Product',
        'Essential Oil/Fat',
        'Flower',
        'Fruit',
        'Fungus',
        'Nut/Seed',
        'Meat/Animal Product',
        'Plant/Vegetable',
        'Sauce/Powder/Dressing',
        'Seafood',
        'Spice',
        'ETC'
    ]

    for index, node in enumerate(nodes):
        x = dict_ingr2count[node]
        if x < 1000:
            size = 5
        elif x < 10000:
            size =  x/300
        elif x < 50000:
            size =  x/800
        elif x < 100000:
            size = x/3000
        elif x < 200000:
            size = x/5000
        else:
            size = 50
        node_sizes+=[size]
        categories+=[dict_ingr2cate[node]]

    trace_node0 = make_trace_nodes(pos, nodes, node_sizes, categories, category_order, category2color)


    Xv1=[]
    Yv1=[]
    label1 = []

    for index, node in enumerate(nodes_fcomps):
        Xv1+=[pos2[index][0]]
        Yv1+=[pos2[index][1]]
        label1+=[node]

    #nodes1 - food-like compounds
    trace_node1=go.Scatter(x=Xv1,
                   y=Yv1,
                   mode='markers',
                   name='food-like compounds',
                   marker=dict(symbol='H',
                                 size=10,
                                 color='green',
                                 line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                   text=label1,
                   hoverinfo='text'
                   )

    Xv2=[]
    Yv2=[]
    label2 = []

    for index, node in enumerate(nodes_dcomps):
        Xv2+=[pos3[index][0]]
        Yv2+=[pos3[index][1]]
        label2+=[node]

    #nodes2 - drug-like compounds
    trace_node2=go.Scatter(x=Xv2,
                   y=Yv2,
                   mode='markers',
                   name='drug-like compounds',
                   marker=dict(symbol='H',
                                 size=10,
                                 color='pink',
                                 line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                   text=label1,
                   hoverinfo='text'
                   )

def circular_layout(graph):
    pos = []

    # radius of the circle
    circle_r = 9
    # center of the circle (x, y)
    circle_x = 0
    circle_y = 0

    for index, node in enumerate(nodes_fcomps):
        # random angle
        alpha = 2 * math.pi * random.random()
        # random radius
        r = circle_r #* math.sqrt(random.random())
        # calculating coordinates
        x = r * math.cos(alpha) + circle_x
        y = r * math.sin(alpha) + circle_y

        fcpos+=[[x,y]]

    return pos

def custom_cate_layout(graph):
    pos_dict = {}
    cate_set = list(set([node[1] for node in graph.nodes(data='category')]))
    Xs = np.random.choice(range(-5,5), len(cate_set), replace=True)
    Ys = np.random.choice(range(-5,5), len(cate_set), replace=True)

    for ix, cate in enumerate(cate_set):
        x = Xs[ix]
        y = Ys[ix]
        for node in graph.nodes(data='category'):
            if node[1] == cate:
                x += round(random.uniform(-0.5,0.5),3)
                y += round(random.uniform(-0.5,0.5),3)
                pos_dict[node[0]] = [x, y]
    return pos_dict

# Embedding
def plot_embedding():
    return
