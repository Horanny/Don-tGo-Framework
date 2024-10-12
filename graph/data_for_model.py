#-*-coding:utf-8-*-

from utils import utils
import json, codecs
import os
import numpy as np
import math
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from matplotlib import pyplot as plt
import gensim
# import pickle5 as pickle

import time

def opt1_fn(deg):
    if len(deg) == 1:
        return [[deg[0], 1]]
    opt_deg = []
    last_deg = deg[0]
    cnt = 1
    for cur_deg in deg[1:-1]:
        if cur_deg == last_deg:
            cnt += 1
        else:
            opt_deg.append([last_deg, cnt])
            last_deg = cur_deg
            cnt = 1
    if deg[-1] == last_deg:
        opt_deg.append([last_deg, cnt+1])
    else:
        opt_deg.append([deg[-1], 1])
    return opt_deg

def compute_k_neighbour(root, graph, all_nodes, k, opt=True):
    degree_list = []
    queue = [root]
    visited = {node:False for node in all_nodes}
    visited[root] = True
    for i in range(k):
        que_len = len(queue)
        if que_len == 0:
            break
        deg_i = []
        for j in range(que_len):
            node = queue.pop(0)
            adj_length = len(graph[node])
            assert adj_length != 0, 'Error at adj_length == 0, where node = %d'%node
            deg_i.append(adj_length)
            for item in graph[node]:
                if not visited[item]:
                    queue.append(item)
                    visited[item] = True
        deg_i = sorted(deg_i)
        if opt:
            deg_i = opt1_fn(deg_i)
        assert deg_i != [], 'Error at deg_i == [], where root = %d, k = %d' % (root, k)
        degree_list.append(deg_i)
    return degree_list

def _create_vectors(graph, all_nodes):
    degrees = {}  # sotre v list of degree
    degrees_sorted = set()  # store degree
    for v in all_nodes:
        degree = len(graph[v])
        degrees_sorted.add(degree)
        if (degree not in degrees):
            degrees[degree] = {}
            degrees[degree]['vertices'] = []
        degrees[degree]['vertices'].append(v)
    degrees_sorted = np.array(list(degrees_sorted), dtype='int')
    degrees_sorted = np.sort(degrees_sorted)

    l = len(degrees_sorted)
    for index, degree in enumerate(degrees_sorted):
        if (index > 0):
            degrees[degree]['before'] = degrees_sorted[index - 1]
        if (index < (l - 1)):
            degrees[degree]['after'] = degrees_sorted[index + 1]

    return degrees

def verifyDegrees(degrees, degree_v_root, degree_a, degree_b):
    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now

def get_vertices(v, degree_v, degrees, n_nodes):
    a_vertices_selected = 2 * math.log(n_nodes, 2)
    vertices = []
    try:
        c_v = 0

        for v2 in degrees[degree_v]['vertices']:
            if (v != v2):
                vertices.append(v2)  # same degree
                c_v += 1
                if (c_v > a_vertices_selected):
                    raise StopIteration

        if ('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if ('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if (degree_b == -1 and degree_a == -1):
            raise StopIteration  # not anymore v
        degree_now = verifyDegrees(degrees, degree_v, degree_a, degree_b)
        # nearest valid degree
        while True:
            for v2 in degrees[degree_now]['vertices']:
                if (v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if (c_v > a_vertices_selected):
                        raise StopIteration

            if (degree_now == degree_b):
                if ('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if ('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']

            if (degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees, degree_v, degree_a, degree_b)

    except StopIteration:
        return list(vertices)

def dist_fn(a,b):
    assert(min(a,b)!=0)
    return max(a,b)/(min(a,b)+0.1)-1

def opt_dist_fn(a,b):
    return (max(a[0]+1,b[0]+1)/min(a[0]+1,b[0]+1)-1)*max(a[1],b[1])

def i_level_graph(all_nodes, similar_node, i, degree_dict):
    # dist_graph = {}
    dist_graph = nx.Graph()
    all_weight = 0
    cnt = 0
    for st in all_nodes:
        if i >= len(degree_dict[st]):
            continue
        in_x = degree_dict[st][i]
        for ed in similar_node[st]:
            if i >= len(degree_dict[ed]):
                continue
            in_y = degree_dict[ed][i]
            dist, _ = fastdtw(in_x, in_y, dist=dist_fn)
            weight = np.power(np.e, -0.5 * dist)
            # dist_graph[st][ed] = weight
            dist_graph.add_edge(st, ed, weight=weight)
            all_weight += weight
            cnt += 1
    return dist_graph, all_weight, cnt

def level_graph(similar_node, degree_dict, k):
    dist_graph = [nx.Graph() for i in range(k)]
    all_weight = [0]*k
    all_cnt = [0]*k
    for st, ed in similar_node.edges:
        length = min(len(degree_dict[st]), len(degree_dict[ed]))
        weight = 0
        for i in range(length):
            in_x = degree_dict[st][i]
            in_y = degree_dict[ed][i]
            dist, _ = fastdtw(np.array(in_x), np.array(in_y), dist=opt_dist_fn)
            weight = np.power(np.e, -(dist+weight))
            dist_graph[i].add_edge(st, ed, weight=weight)
            all_weight[i] += weight
            all_cnt[i] += 1
    return dist_graph, all_weight, all_cnt

def compute_degree_graph(all_nodes, undirected_graph, k):
    # compute multi-level degree
    # k = 8
    degree_dict = {}
    for key in all_nodes:
        if len(undirected_graph[key])!=0:
            deg = compute_k_neighbour(key, undirected_graph, all_nodes, k)
            degree_dict[key] = deg
    # with ProcessPoolExecutor(8) as threadpool:
    #     future_list = []
    #     for key in all_nodes:
    #         if len(undirected_graph[key]) != 0:
    #             future = threadpool.submit(compute_k_neighbour, key, undirected_graph, all_nodes, k, True)
    #             future_list.append([key, future])
    #     for key, item in future_list:
    #         result = item.result()
    #         degree_dict[key] = result

    print('degree_graph computation finish')

    return degree_dict

def compute_similar_node(all_nodes, undirected_graph):
    # OPT2
    degrees = _create_vectors(undirected_graph, all_nodes)
    # similar_node = {}
    similar_node = nx.Graph()
    for v in all_nodes:
        if len(undirected_graph[v]) == 0: continue
        nbs = get_vertices(v, len(undirected_graph[v]), degrees, len(all_nodes))
        # similar_node[v] = nbs
        for item in nbs:
            similar_node.add_edge(v, item)
    print('OPT2 computation finish')

    return similar_node

def struc2vec_graph():
    tagpath = utils.get_line_tag_path()
    with open(tagpath, 'r') as f:
        raw_tag = f.readlines()
    line_tag = [int(item) for item in raw_tag]
    length = 6#len(line_tag)
    start = 0
    for cur_hour in range(start, length):
        graph_path = os.path.join(utils.get_dynamic_graph_dir(), 'dyn_graph_%d.gpickle' % cur_hour)
        # with open(graph_path, 'r') as f:
        #     raw_data = f.readlines()
        #     undirected_graph = json.loads(raw_data[0])
        undirected_graph = nx.read_gpickle(graph_path)
        print('undirected graph %d is loaded' % cur_hour)

        all_nodes = list(filter(lambda x: len(undirected_graph[x])!=0, undirected_graph.nodes))

        k = 8
        '''degree graph'''
        degree_dict = compute_degree_graph(all_nodes, undirected_graph, k)
        '''similar nodes'''
        similar_node = compute_similar_node(all_nodes, undirected_graph)
        '''dist graph'''
        nx_graph_list, sum_weight, cnt_weight = level_graph(similar_node, degree_dict, k)

        # save
        dist_graph_path = os.path.join(utils.get_dist_dir(), 'dist_graph_list_%d.json' % cur_hour)
        dist_graph_list = []
        for graph in nx_graph_list:
            dist_graph_list.append(list(graph.edges.data('weight')))
        with codecs.open(dist_graph_path, 'w', 'utf-8') as f:
            json.dump(dist_graph_list, f, ensure_ascii=False)
            f.write('\n')
        print('dist_graph_list save')

        # between layers
        between_graph_list = []
        for i in range(1, k - 1):
            dist_graph = nx_graph_list[i]
            between_graph = {n: 0 for n in dist_graph.nodes}
            avg = sum_weight[i]/cnt_weight[i]
            for u, v, w in dist_graph.edges.data('weight'):
                if w > avg:
                    between_graph[u] += 1
            for u in dist_graph.nodes:
                between_graph[u] = np.log(np.e+between_graph[u])
            between_graph_list.append(between_graph)

        # save
        between_graph_path = os.path.join(utils.get_between_dir(), 'between_graph_list_%d.json' % cur_hour)
        with codecs.open(between_graph_path, 'w', 'utf-8') as f:
            json.dump(between_graph_list, f, ensure_ascii=False)
            f.write('\n')
        print('between_graph_list save')

def tsne_projection(x, k,perplexity=50, n_iter=5000):
    n, d = x.shape
    tsne = TSNE(n_components=k, init='pca', perplexity=perplexity, n_iter=n_iter)
    res = tsne.fit_transform(x)
    # x_centered = x - x_mean.unsqueeze(1).repeat([1,n])
    # u, s, v = torch.svd(x_centered)
    # u_reduced = u[:, :k]
    # res_centered = torch.mm(u_reduced.transpose(1,0), x_centered)
    # res = res_centered + torch.mm(u_reduced.transpose(1,0), x_mean.unsqueeze(1)).repeat([1,n])
    return res

def read_k_degree_graph(h, is_min):
    # build in-degree graph
    graph_path = os.path.join(utils.get_dynamic_graph_dir(), 'dyn_graph_%d.gpickle' % h)
    undirected_graph = nx.read_gpickle(graph_path)
    # with open(graph_path, 'rb') as f:
    #     undirected_graph = pickle.load(f)
    print('undirected graph %d is loaded' % h)

    '''minimal node set is from struc2vec embedding result'''
    if is_min == 0:
        all_nodes = list(filter(lambda x: len(undirected_graph[x])!=0, undirected_graph.nodes))
    elif is_min == 1:
        wv = gensim.models.Word2Vec.load(os.path.join(utils.get_struc2vec_dir(), 'model_%d.ckpt' % h)).wv
        all_nodes = [int(n) for n in wv.key_to_index]
    else:
        wv = gensim.models.Word2Vec.load(os.path.join(utils.get_struc2vec_dir(), 'model_%d.ckpt' % h)).wv
        nodes1 = [int(n) for n in wv.key_to_index]
        nodes2 = list(filter(lambda x: len(undirected_graph[x]) != 0, undirected_graph.nodes))
        all_nodes = list(set(nodes1)&set(nodes2))

    # read
    dist_graph_path = os.path.join(utils.get_dist_dir(), 'dist_graph_list_%d.json' % h)
    with codecs.open(dist_graph_path, 'r') as f:
        raw_data = f.readlines()
        graph_list = json.loads(raw_data[0])
    dist_graph_list = []
    for item in graph_list:
        graph = nx.Graph()
        for u,v,w in item:
            graph.add_edge(int(u),int(v),weight=int(w))
        dist_graph_list.append(graph)
    print('dist_graph_list read')

    # read
    between_path = os.path.join(utils.get_between_dir(), 'between_graph_list_%d.json' % h)
    with codecs.open(between_path, 'r') as f:
        raw_data = f.readlines()
        tmp = json.loads(raw_data[0])
    between_graph_list = []
    for between_graph in tmp:
        between_graph_list.append({int(key): val for key, val in between_graph.items()})
    '''TODO: modifying after retrain!!!'''
    between_graph_list = [{n: 0 for n in dist_graph_list[0].nodes}] + between_graph_list[1:] + [{n: 0 for n in dist_graph_list[-1].nodes}]
    print('between_graph_list read')

    return undirected_graph, dist_graph_list, between_graph_list, all_nodes

def dyngem_graph(cur_hour):
    '''deprecated'''
    graph_path = os.path.join(utils.get_graph_dir(), 'undirected_graph_%d.json' % cur_hour)
    with open(graph_path, 'r') as f:
        raw_data = f.readlines()
        undirected_graph = json.loads(raw_data[0])
    print('undirected graph %d is loaded' % cur_hour)

    all_nodes = list(undirected_graph.keys())
    node2idx = {node:i for i, node in enumerate(all_nodes)}

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(len(all_nodes)))
    for root in undirected_graph.keys():
        root_idx = node2idx[root]
        neigh_set = undirected_graph[root]
        for neigh in neigh_set.keys():
            neigh_idx = node2idx[neigh]
            nx_graph.add_edge(root_idx, neigh_idx)

    return nx_graph

def dynamic_nx_graph(is_weigth=False):
    tagpath = utils.get_line_tag_path()
    with open(tagpath, 'r') as f:
        raw_tag = f.readlines()
    line_tag = [int(item) for item in raw_tag]
    pgpath = utils.get_pg_path()
    f = open(pgpath, 'r')
    length = len(line_tag)
    cnt = 0
    all_nodes = {}
    node_num = 0
    for cur_hour in range(length - 1):
        cnt, line_data = utils.read_raw_data(f, line_tag[cur_hour], line_tag[cur_hour + 1], cnt)
        nx_graph = nx.Graph()
        for item in line_data:
            gbid = item[1]
            tgbid = item[2]
            intimacy = item[4] if is_weigth else 1
            if gbid not in all_nodes:
                all_nodes[gbid] = node_num
                node_num += 1
            if tgbid not in all_nodes:
                all_nodes[tgbid] = node_num
                node_num += 1
            nx_graph.add_edge(all_nodes[gbid], all_nodes[tgbid], weight=intimacy)
        nx_graph.add_nodes_from(range(len(all_nodes.keys())))
        save_path = os.path.join(utils.get_dynamic_graph_dir(), 'dyn_graph_%d.gpickle'%cur_hour)
        nx.write_gpickle(nx_graph, save_path)
        print('%s save'%save_path)
    f.close()
    all_nodes_path = utils.get_all_nodes_path()
    with codecs.open(all_nodes_path, 'w', 'utf-8') as outf:
        json.dump(all_nodes, outf, ensure_ascii=False)
        outf.write('\n')
    print('%s save'%all_nodes_path)

def build_undirected_graph():
    tagpath = utils.get_line_tag_path()
    with open(tagpath, 'r') as f:
        raw_tag = f.readlines()
    line_tag = [int(item) for item in raw_tag]
    pgpath = utils.get_pg_path()
    f = open(pgpath, 'r')
    length = len(line_tag)
    cnt = 0
    for cur_hour in range(length-1):
        cnt, line_data = utils.read_raw_data(f, line_tag[cur_hour], line_tag[cur_hour + 1], cnt)
        '''add all operation here'''
        undirected_graph = {}
        for item in line_data:
            gbid = item[1]
            tgbid = item[2]
            intimacy = item[4]
            if gbid not in undirected_graph:
                undirected_graph[gbid] = {}
            if tgbid not in undirected_graph:
                undirected_graph[tgbid] = {}
            undirected_graph[gbid][tgbid] = intimacy
            undirected_graph[tgbid][gbid] = intimacy

        graph_path = os.path.join(utils.get_graph_dir(), 'undirected_graph_%d.json' % cur_hour)
        with codecs.open(graph_path, 'w+', 'utf-8') as outf:
            json.dump(undirected_graph, outf, ensure_ascii=False)
            outf.write('\n')
        print('undirected graph %d is saved' % cur_hour)
    f.close()

def projection(align_path, save_path, cur_hour, hours, delta_time):
    # cur_hour = 3
    # hours = 6
    # delta_time = 3
    for j in range(cur_hour, hours, delta_time):
        np_data = []
        for i in range(j, j + delta_time):
            undirected_graph, dist_graph_list, between_graph_list, all_nodes = read_k_degree_graph(i)
            name = os.path.join(align_path, 'dynAERNN_%d.npy' % (i))
            emb_h1_ali = np.load(name)
            np_data.append(emb_h1_ali[all_nodes])

        np_data_cat = np.concatenate(np_data, axis=0)
        np_proj = tsne_projection(np_data_cat, 2, 100)
        y_pred = DBSCAN(eps=5, min_samples=20).fit_predict(np_proj)
        np.save(os.path.join(save_path, 'tsne_proj_%d.npy' % j), np_proj)
        np.save(os.path.join(save_path, 'tsne_pred_%d.npy' % j), y_pred)
        print('cluster numbers: %d' % (max(y_pred) + 1))
        plt.figure()
        plt.scatter(np_proj[:, 0], np_proj[:, 1], c=y_pred)
        plt.savefig(os.path.join(save_path, 'tsne_proj_%d.jpg'%j))
        plt.show()

def analysis(cur_hour, tsne_path, delta_time, is_sorted):
    # np_proj = np.load(os.path.join(utils.get_tsne_of_dyn_dir(),
    #                                'tsne_proj_%d.npy' % cur_hour))
    y_pred = np.load(os.path.join(tsne_path, 'tsne_pred_%d.npy' % cur_hour))
    '''merge delta graphs into one graph'''
    graph_length = [0]
    bias_list = [0]
    all_node_list = []
    merge_graph = nx.Graph()
    bias = 0
    for t in range(cur_hour, cur_hour+delta_time):
        undirected_graph, dist_graph_list, between_graph_list, all_nodes = read_k_degree_graph(t)
        '''only for dynAERNN'''
        # all_nodes = list(filter(lambda x: x != 11140, all_nodes))
        if is_sorted: all_nodes = sorted(all_nodes)
        all_nodes_length = len(all_nodes)
        graph_length.append(all_nodes_length)
        all_node_list += [n+bias for n in all_nodes]
        for u,v,w in undirected_graph.edges.data('weight'):
            merge_graph.add_edge(u+bias, v+bias, weight=w)
        # bias += all_nodes_length
        bias = max(all_node_list)+1
        bias_list.append(bias)
    '''construct subgraph of each cluster'''
    subgraph_list = []
    node_set = []
    max_class = max(y_pred)+1
    for i in range(max_class):
        node_list = (y_pred==i).nonzero()[0]
        node_list = list(np.array(all_node_list)[node_list])
        node_set.append(node_list)
        # subgraph = merge_graph.subgraph(node_list).copy()
        # subgraph = nx.Graph()
        # subgraph.add_edges_from(merge_graph.edges(node_list))
        # subgraph = nx.edge_subgraph(merge_graph, merge_graph.edges(node_list))
        subgraph = merge_graph.subgraph(node_list)
        subgraph_list.append(subgraph)

    print('\nmax class = %d\n'%max_class)

    connected_components = []
    similar_dist = []
    same_time = []
    k = 3
    for i in range(max_class):
        subgraph = subgraph_list[i]
        '''the more connected components, the less nodes are related'''
        cc_num = nx.algorithms.components.number_connected_components(subgraph)
        connected_components.append(cc_num)
        print('connected components: %d' % cc_num)
        '''same time'''
        time_cnt = [0]*delta_time
        for node in subgraph.nodes:
            for t in range(delta_time):
                if node>=bias_list[t] and node<bias_list[t+1]:
                    time_cnt[t] += 1
                    break
        time_span = np.array(time_cnt)/np.array(graph_length[1:])
        same_time.append(time_span)
        print_str = 'ratio of being in same time span: '
        for ts in time_span:
            print_str += ' %f' % ts
        print(print_str)
        time_cons = np.array(time_cnt)/len(subgraph.nodes)
        print_str = 'ratio of time construction: '
        for ts in time_cons:
            print_str += ' %f' % ts
        print(print_str)
        '''dist between similar nodes'''
        degree_dict = compute_degree_graph(subgraph.nodes(), subgraph, k)
        similar_node = compute_similar_node(subgraph.nodes(), subgraph)
        for item in similar_node.nodes:
            if item not in degree_dict:
                degree_dict[item] = [[]]
            for _item in degree_dict[item]:
                _item.append([0,0])
        assert len(degree_dict) == len(similar_node.nodes) and len(similar_node.nodes) >0
        dist_graph, sum_weight, cnt_weight = level_graph(similar_node, degree_dict, k)
        distance = np.array(sum_weight)/np.array(cnt_weight)
        similar_dist.append(distance)
        print_str = 'distance between similar: '
        for dis in distance:
            print_str+=' %f'%dis
        print(print_str+'\n')
    print('finish\n')

def graph_sparsity(st, ed):
    sparsity = []
    for t in range(st, ed):
        # build in-degree graph
        graph_path = os.path.join(utils.get_dynamic_graph_dir(), 'dyn_graph_%d.gpickle' % t)
        undirected_graph = nx.read_gpickle(graph_path)
        # with open(graph_path, 'rb') as f:
        #     undirected_graph = pickle.load(f)
        print('undirected graph %d is loaded' % t)
        all_nodes = list(filter(lambda x: len(undirected_graph[x]) != 0, undirected_graph.nodes))
        n_node = len(all_nodes)
        n_edge = len(undirected_graph.edges)
        t_spa = n_edge/n_node/n_node
        sparsity.append(t_spa)
        print('nodes: %d, edges: %d, sparsity at %d: %f'%(n_node, n_edge, t, t_spa))
    print('graph sparsity: %f\n'%(sum(sparsity)/len(sparsity)))

def generate_event_list(st, ed):
    with codecs.open(utils.get_all_nodes_path(), 'r') as f:
        raw_data = f.readlines()
        node_map = json.loads(raw_data[0])

    tagpath = utils.get_line_tag_path()
    with open(tagpath, 'r') as f:
        raw_tag = f.readlines()
    line_tag = [int(item) for item in raw_tag]
    length = len(line_tag)

    pgpath = utils.get_pg_path()
    event_path = utils.get_event_list_dir()
    with open(pgpath, 'r') as f:
        cnt = 0
        for i in range(st, ed):
            undirected_graph, dist_graph_list, between_graph_list, all_nodes = read_k_degree_graph(i, is_min=0)
            events = {_n: [] for _n in all_nodes}
            cnt, line_data = utils.read_raw_data(f, line_tag[i], line_tag[i + 1], cnt)
            for item in line_data:
                gbid = node_map[item[1]]
                tgbid = node_map[item[2]]
                evt_type = item[3]
                if events[gbid] != []:
                    g_last_evt = events[gbid][-1]
                    if g_last_evt != evt_type:
                        events[gbid].append(evt_type)
                else:
                    events[gbid].append(evt_type)
                if events[tgbid] != []:
                    tg_last_evt = events[tgbid][-1]
                    if tg_last_evt != evt_type:
                        events[tgbid].append(evt_type)
                else:
                    events[tgbid].append(evt_type)
            with codecs.open(os.path.join(event_path, 'event_list_%d.json'%i), 'w', 'utf-8') as ff:
                json.dump(events, ff, ensure_ascii=False)
                ff.write('\n')

def generate_full_event_list(st, ed):
    with codecs.open(utils.get_all_nodes_path(), 'r') as f:
        raw_data = f.readlines()
        node_map = json.loads(raw_data[0])

    tagpath = utils.get_line_tag_path()
    with open(tagpath, 'r') as f:
        raw_tag = f.readlines()
    line_tag = [int(item) for item in raw_tag]
    length = len(line_tag)

    pgpath = utils.get_pg_path()
    event_path = utils.get_full_event_list_dir()
    with open(pgpath, 'r') as f:
        cnt = 0
        for i in range(st, ed):
            undirected_graph, dist_graph_list, between_graph_list, all_nodes = read_k_degree_graph(i, is_min=0)
            events = {_n: [] for _n in all_nodes}
            cnt, line_data = utils.read_raw_data(f, line_tag[i], line_tag[i + 1], cnt)
            for item in line_data:
                gbid = node_map[item[1]]
                tgbid = node_map[item[2]]
                evt_type = item[3]
                events[gbid].append(evt_type)
                events[tgbid].append(evt_type)
            with codecs.open(os.path.join(event_path, 'event_list_%d.json'%i), 'w', 'utf-8') as ff:
                json.dump(events, ff, ensure_ascii=False)
                ff.write('\n')

def dynamic_nodes(save_path, is_sort=False):
    st = 3
    ed = 6
    delta_t = 3
    dynamic = 0
    total = 0
    # all nodes
    nodes = []
    for i in range(st, ed):
        undirected_graph, dist_graph_list, between_graph_list, all_nodes = read_k_degree_graph(i)
        if is_sort: all_nodes = sorted(all_nodes)
        nodes.append(all_nodes)
    player_length = [0] + list(map(len, nodes))
    for i in range(2, delta_t + 1):
        player_length[i] += player_length[i - 1]

    # projection result
    y_pred = []
    y_path = utils.get_tsne_of_dyn_dir()
    _y = np.load(os.path.join(save_path, 'tsne_pred_%d.npy' % st))
    for i, bias in enumerate(player_length[:-1]):
        _map = {item: _y[_i + bias] for _i, item in enumerate(nodes[i])}
        y_pred.append(_map)

    assert player_length[-1] == sum([len(item) for item in y_pred])

    view_label = []
    intersection = []
    for i in range(0, delta_t-1):
        _node = list(filter(lambda x: x in nodes[i], nodes[i+1]))
        intersection.append(_node)
        view_label += [int(y_pred[i][item] == y_pred[i+1][item]) for item in _node]

    dynamic += sum(view_label)
    total += sum(map(len,intersection))
    print(dynamic/total)

def _si_foo(node, dists, k):
    neighbor = [[key,val] for key, val in dists[0][node].items()]
    k_hop = [0] * k
    k_cnt = [1e-5] * k
    for i in range(k):
        k_cnt[i] += len(neighbor)
        k_hop[i] += sum(map(lambda x:x[1]['weight'], neighbor))
        _neigh = list(filter(lambda x:x not in dists[i], neighbor))
        if i<k-1:
            neighbor = np.concatenate(list(map(lambda x: np.array(list(dists[i][x[0]].items())), _neigh)), axis=0).tolist()
    return [weight / cnt for weight, cnt in zip(k_hop, k_cnt)]

def _bfs(data, indices, indptr, node, k):
    pass

def structural_identity(nodes, dists, betweens):
    k=3
    result = {}
    # with ProcessPoolExecutor(8) as threadpool:
    #     future_list = []
    #     for node in nodes:
    #         future = threadpool.submit(_si_foo, node, dists, k)
    #         future_list.append([node,future])
    #     for item in future_list:
    #         _res = item.result()
    #         result[_res[0]] = _res[1]
    sparse_list = []
    for i in range(k):
        sparse = nx.to_scipy_sparse_matrix(dists[0], nodelist=nodes)
        data = sparse.data
        indices = sparse.indices
        indptr = sparse.indptr
        sparse_list.append(sparse)
    for node in nodes:
        st = time.time()
        result[node] = _si_foo(node, dists, k)
        print(time.time()-st)
    return result

# @nb.jit(nopython=True)
# def structural_identity(nodes, dists, betweens, k=1):
#     result = {}
#     for node in nodes:
#         neighbor = []
#         for key,val in dists[0][node].items():
#             neighbor.append([key,val])
#         k_hop = [0]*k
#         k_cnt = [1e-5]*k
#         _res = []
#         for i in range(k):
#             for key,val in neighbor:
#                 k_hop[i] += val['weight']
#                 k_cnt[i] += 1
#             for key,val in neighbor:
#                 neighbor.pop(0)
#                 for _key,_val in dists[i+1][key].items():
#                     neighbor.append([_key,_val['weight']])
#             _res.append(k_hop[i]/k_cnt[i])
#         result[node] = _res
#     return result

def degree(nodes, graph):
    result = {}
    # graph = graph.subgraph(nodes)
    for node in nodes:
        result[node] = len(graph[node])
    return result

def page_rank(nodes, graph):
    '''not in dist graph, but in undirected graph'''
    # return nx.pagerank(graph.subgraph(nodes))
    pr = nx.pagerank(graph)
    return dict(filter(lambda x:x[0] in nodes, pr.items()))

def betweenness(nodes, graph):
    bc = nx.betweenness_centrality(graph.subgraph(nodes))
    return dict(filter(lambda x: x[0] in nodes, bc.items()))

def leverage(nodes, graph):
    result = {}
    # graph = graph.subgraph(nodes)
    for node in nodes:
        degv = len(graph[node])
        degu = [len(graph[n]) for n in graph[node]]
        _each = [(degv-deg)/(degv+deg) for deg in degu]
        _res = sum(_each)/degv
        result[node] = _res
    return result

def average_nearest_neighbors(nodes, graph):
    result = {}
    # graph = graph.subgraph(nodes)
    for node in nodes:
        result[node] = sum([item['weight'] for item in graph[node].values()])/len(graph[node])
    return result

def within_module_degree(nodes, graph):
    result = {}
    # graph = graph.subgraph(nodes)
    degs = np.array([len(graph[item]) for item in graph])
    avg = np.average(degs)
    dev = np.sqrt(np.average((degs-avg)**2))
    for node in nodes:
        result[node] = (len(graph[node])-avg)/dev
    return result

def closeness(nodes, graph):
    cn = nx.closeness_centrality(graph.subgraph(nodes))
    return dict(filter(lambda x:x[0] in nodes, cn.items()))

def t_foo(arr, n):
    for i,item in enumerate(arr):
        if item >n:
            return i-1

def t_fun(_n, time_stamp):
    t = int(np.round((_n-int(_n))*100))-time_stamp
    assert t>=0
    return t

def all_network_metrics(tsne_path,save_path,cur_hour, hours, delta_t=3, is_sorted=False):
    for time_stamp in range(cur_hour, hours, delta_t):
        # bias_list = [0]
        # bias = 0
        # y_pred = np.load(os.path.join(tsne_path, 'tsne_pred_%d.npy' % time_stamp), allow_pickle=True).item()
        # y_pred = np.load(os.path.join(tsne_path, 'tsne_xmeans_%d.npy' % time_stamp), allow_pickle=True).item()
        deg_list = []
        pr_list = []
        bc_list = []
        lc_list = []
        wmd_list = []
        cn_list = []
        nodes = []
        for t in range(time_stamp, time_stamp+delta_t):
            print('Time stamp = %d'%t)
            undirected_graph, dist_graph_list, between_graph_list, all_nodes = read_k_degree_graph(t, is_min=2)
            if is_sorted: all_nodes = sorted(all_nodes)
            nodes.append(all_nodes)
            # nodes += [bias+n for n in all_nodes]
            # bias = max(nodes) + 1
            # bias_list.append(bias)
            deg = degree(all_nodes, undirected_graph)
            deg_list.append(deg)
            print('degree computed at %d' % t)
            pr = page_rank(all_nodes, undirected_graph)
            pr_list.append(pr)
            print('page rank computed at %d' % t)
            bc = betweenness(all_nodes, undirected_graph)
            bc_list.append(bc)
            print('betweenness centrality computed at %d' % t)
            lc = leverage(all_nodes, undirected_graph)
            lc_list.append(lc)
            print('leverage centrality computed at %d' % t)
            wmd = within_module_degree(all_nodes, undirected_graph)
            wmd_list.append(wmd)
            print('within module degree computed at %d' % t)
            cn = closeness(all_nodes, undirected_graph)
            cn_list.append(cn)
            print('closeness computed at %d' % t)
        with codecs.open(os.path.join(save_path, 'deg_list_%d.json'%time_stamp), 'w', 'utf-8') as f:
            json.dump(deg_list, f, ensure_ascii=False)
            f.write('\n')
        with codecs.open(os.path.join(save_path, 'pr_list_%d.json'%time_stamp), 'w', 'utf-8') as f:
            json.dump(pr_list, f, ensure_ascii=False)
            f.write('\n')
        with codecs.open(os.path.join(save_path, 'bc_list_%d.json'%time_stamp), 'w', 'utf-8') as f:
            json.dump(bc_list, f, ensure_ascii=False)
            f.write('\n')
        with codecs.open(os.path.join(save_path, 'lc_list_%d.json'%time_stamp), 'w', 'utf-8') as f:
            json.dump(lc_list, f, ensure_ascii=False)
            f.write('\n')
        with codecs.open(os.path.join(save_path, 'wmd_list_%d.json'%time_stamp), 'w', 'utf-8') as f:
            json.dump(wmd_list, f, ensure_ascii=False)
            f.write('\n')
        with codecs.open(os.path.join(save_path, 'cn_list_%d.json'%time_stamp), 'w', 'utf-8') as f:
            json.dump(cn_list, f, ensure_ascii=False)
            f.write('\n')

def network_metrics(tsne_path,save_path,list_path,cur_hour, hours, delta_t=3, is_sorted=False):
    for time_stamp in range(cur_hour, hours, delta_t):
        # bias_list = [0]
        # bias = 0
        # y_pred = np.load(os.path.join(tsne_path, 'tsne_pred_%d.npy' % time_stamp), allow_pickle=True).item()
        y_pred = np.load(os.path.join(tsne_path, 'tsne_xmeans_%d.npy' % time_stamp), allow_pickle=True).item()
        deg_list = []
        pr_list = []
        bc_list = []
        lc_list = []
        wmd_list = []
        cn_list = []
        nodes = []
        for t in range(time_stamp, time_stamp+delta_t):
            print('Time stamp = %d'%t)
            undirected_graph, dist_graph_list, between_graph_list, all_nodes = read_k_degree_graph(t, is_min=2)
            if is_sorted: all_nodes = sorted(all_nodes)
            nodes.append(all_nodes)
        #     # nodes += [bias+n for n in all_nodes]
        #     # bias = max(nodes) + 1
        #     # bias_list.append(bias)
        #     deg = degree(all_nodes, undirected_graph)
        #     deg_list.append(deg)
        #     print('degree computed at %d' % t)
        #     pr = page_rank(all_nodes, undirected_graph)
        #     pr_list.append(pr)
        #     print('page rank computed at %d'%t)
        #     bc = betweenness(all_nodes, undirected_graph)
        #     bc_list.append(bc)
        #     print('betweenness centrality computed at %d' % t)
        #     lc = leverage(all_nodes, undirected_graph)
        #     lc_list.append(lc)
        #     print('leverage centrality computed at %d' % t)
        #     wmd = within_module_degree(all_nodes, undirected_graph)
        #     wmd_list.append(wmd)
        #     print('within module degree computed at %d' % t)
        #     cn = closeness(all_nodes, undirected_graph)
        #     cn_list.append(cn)
        #     print('closeness computed at %d' % t)
        with codecs.open(os.path.join(list_path, 'deg_list_%d.json'%time_stamp), 'r') as f:
            raw_data = f.readlines()
            deg_list = json.loads(raw_data[0])
        with codecs.open(os.path.join(list_path, 'pr_list_%d.json'%time_stamp), 'r') as f:
            raw_data = f.readlines()
            pr_list = json.loads(raw_data[0])
        with codecs.open(os.path.join(list_path, 'bc_list_%d.json'%time_stamp), 'r') as f:
            raw_data = f.readlines()
            bc_list = json.loads(raw_data[0])
        with codecs.open(os.path.join(list_path, 'lc_list_%d.json' % time_stamp), 'r') as f:
            raw_data = f.readlines()
            lc_list = json.loads(raw_data[0])
        with codecs.open(os.path.join(list_path, 'wmd_list_%d.json'%time_stamp), 'r') as f:
            raw_data = f.readlines()
            wmd_list = json.loads(raw_data[0])
        with codecs.open(os.path.join(list_path, 'cn_list_%d.json'%time_stamp), 'r') as f:
            raw_data = f.readlines()
            cn_list = json.loads(raw_data[0])
        max_pred = int(max(y_pred.values()))
        for i in range(max_pred+1):
            print('cluster %d: ' % i)
            # _node_idx = (y_pred==i).nonzero()[0]
            # _node_list = list(np.array(nodes)[_node_idx])
            # _time_list = [t_foo(bias_list, _n) for _n in _node_list]
            # _node_list = [_n-bias_list[_t] for _t,_n in zip(_time_list,_node_list)]
            _node_idx = (np.array(list(y_pred.values())) == i).nonzero()[0]
            _raw_node = np.array(list(y_pred.keys()))[_node_idx]
            _raw_node = list(filter(lambda x: int(x) in nodes[t_fun(x, time_stamp)], _raw_node))
            _node_list = [str(int(_n)) for _n in _raw_node]
            _time_list = [t_fun(_n, time_stamp) for _n in _raw_node]
            c_deg = np.array([deg_list[_t][_n] for _t,_n in zip(_time_list,_node_list)])
            np.save(os.path.join(save_path, 'degree_%d.npy'%i), c_deg)
            print('degree=%f' % np.var(c_deg))
            c_pr = np.array([pr_list[_t][_n] for _t,_n in zip(_time_list,_node_list)])
            np.save(os.path.join(save_path, 'page_rank_%d.npy' % i), c_pr)
            print('page rank=%f' % np.var(c_pr))
            c_bc = np.array([bc_list[_t][_n] for _t,_n in zip(_time_list,_node_list)])
            np.save(os.path.join(save_path, 'betweenness_%d.npy' % i), c_bc)
            print('betweenness=%f' % np.var(c_bc))
            c_lc = np.array([lc_list[_t][_n] for _t,_n in zip(_time_list,_node_list)])
            np.save(os.path.join(save_path, 'leverage_%d.npy' % i), c_lc)
            print('leverage=%f' % np.var(c_lc))
            c_wmd = np.array([wmd_list[_t][_n] for _t,_n in zip(_time_list,_node_list)])
            np.save(os.path.join(save_path, 'within_module_degree_%d.npy' % i), c_wmd)
            print('within module degree=%f' % np.var(c_wmd))
            c_cn = np.array([cn_list[_t][_n] for _t,_n in zip(_time_list,_node_list)])
            np.save(os.path.join(save_path, 'closeness_%d.npy' % i), c_cn)
            print('closeness=%f' % np.var(c_cn))
            c_ts = np.array(_time_list)
            np.save(os.path.join(save_path, 'time_distribution_%d.npy' % i), c_ts)
            print('time distribution=%f' % np.var(c_ts))

def cluster_sort(k,path):
    cluster = []
    for i in range(k):
        arr = np.load(os.path.join(path, 'betweenness_%d.npy'%i))
        cluster.append([i,len(arr)])
    cluster = sorted(cluster, key=lambda x:x[1], reverse=True)
    print([n[0] for n in cluster])

def xmeans_fun(np_data_cat, shape, init_num=5):
    initial_centers = kmeans_plusplus_initializer(np_data_cat, init_num).initialize()
    xmeans_instance = xmeans(np_data_cat, initial_centers, 20)
    xmeans_instance.process()
    cluster_member = xmeans_instance.get_clusters()
    y_pred = np.zeros(shape)
    for cid, item in enumerate(cluster_member):
        for _n in item:
            y_pred[_n] = cid
    return y_pred

if __name__ == '__main__':
    cur_hour = 0
    hours = 24
    delta_time = 3
    # build_undirected_graph()
    # dynamic_nx_graph()
    # struc2vec_graph()
    # projection(utils.get_aernn_result_dir(), utils.get_tsne_of_dyn_dir())
    '''struc2vec analysis'''
    # analysis(3, utils.get_tsne_of_struc2vec_dir(), 3, False)
    '''dynAERNN analysis'''
    # analysis(3, utils.get_tsne_of_dyn_dir(), 3, True)
    '''reproduction analysis'''
    # analysis(3, utils.get_reproduction_tsne_dir(), 3, False)
    '''sparsity'''
    # graph_sparsity(0, 26)
    '''event list'''
    # generate_event_list(0,26)
    # generate_full_event_list(0, 26)
    '''dynamic nodes'''
    # dynamic_nodes(utils.get_tsne_of_struc2vec_dir())
    # dynamic_nodes(utils.get_reproduction_tsne_dir())
    # dynamic_nodes(utils.get_tsne_of_dyn_dir())
    '''metrics'''
    # with ProcessPoolExecutor(8) as threadpool:
    #         future_list = []
    #         for t in range(cur_hour, hours, delta_time):
    #             future = threadpool.submit(all_network_metrics, utils.get_tsne_of_struc2vec_dir(),
    #                                 utils.get_metrics_struc2vec_dir(), t, t+delta_time, is_sorted=False)
    #             future_list.append([t, future])
    #         for t, future in future_list:
    #             future.result()

    # with ProcessPoolExecutor(3) as threadpool:
    #     future1 = threadpool.submit(all_network_metrics, utils.get_reproduction_tsne_dir(), utils.get_metrics_reproduction_dir(), 3, 6, is_sorted=False)
    #     future2 = threadpool.submit(all_network_metrics, utils.get_tsne_of_struc2vec_dir(), utils.get_metrics_struc2vec_dir(), 3, 6, is_sorted=False)
    #     future3 = threadpool.submit(all_network_metrics, utils.get_tsne_of_dyn_dir(), utils.get_metrics_dynAERNN_dir(), 3, 6, is_sorted=True)
    #     future1.result()
    #     future2.result()
    #     future3.result()

    # network_metrics(utils.get_reproduction_tsne_dir(), utils.get_metrics_reproduction_dir(), utils.get_metrics_reproduction_dir(), 3, 6, is_sorted=False)
    # network_metrics(utils.get_tsne_of_struc2vec_dir(), utils.get_metrics_struc2vec_dir(), utils.get_metrics_struc2vec_dir(), 3, 6, is_sorted=False)
    # network_metrics(utils.get_tsne_of_dyn_dir(), utils.get_metrics_dynAERNN_dir(), utils.get_metrics_dynAERNN_dir(), 3, 6, is_sorted=True)
    '''sort cluster id'''
    cluster_sort(8, utils.get_metrics_reproduction_dir())
    cluster_sort(9, utils.get_metrics_struc2vec_dir())
    cluster_sort(20, utils.get_metrics_dynAERNN_dir())