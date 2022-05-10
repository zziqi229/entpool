import heapq
import queue
import random
import sys
import numpy as np
import math
from queue import Queue
import copy

nnn = 0


def get_id():
    i = 0
    while True:
        yield i
        i += 1


class Graph:
    def __init__(self, path=None, edges=None, n=-1):
        if (edges is not None):
            t = dict()
            for e in edges:
                et = (e[0], e[1])
                t[et] = t.get(et, 0) + 1
            edges = t
        if (path is not None):
            edges = dict()
            with open(path) as f:
                n, m = f.readline().split()
                n, m = int(n), int(m)
                for i in range(m):
                    a, b = f.readline().split()
                    a, b = int(a), int(b)
                    if (a != b):
                        edges[(a, b)] = edges.get((a, b), 0) + 1
        degree = np.zeros(n)
        for k, w in edges.items():
            a, b = k
            degree[a] += w
            degree[b] += w
        m = degree.sum() / 2
        self.adj_table = [dict() for i in range(n)]
        self.n = n
        self.adj_matrix = np.zeros((n, n))
        for k, w in edges.items():
            a, b = k
            w /= math.sqrt(degree[a] * degree[b])
            if (w > 0):
                self.adj_table[a][b] = self.adj_table[b][a] = w
                self.adj_matrix[a][b] = self.adj_matrix[b][a] = w


class tree_node:
    def __init__(self, ID, vol=0, g=0, parent=None, children=None, partition=None, child_cut=0., h=0, d=0, adj_table=None):
        self.ID = ID
        self.vol = vol
        self.g = g
        self.parent = parent
        self.children = children
        if self.children is None:
            self.children = set()
        self.partition = partition
        if self.partition is None:
            self.partition = list()
        self.child_cut = child_cut
        self.adj_table = adj_table
        if self.adj_table is None:
            self.adj_table = dict()
        self.h = h
        self.d = d
        self.merged = False
        self.removed = False

        self.entory = 0

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())


class Tree:
    def __init__(self, G):
        self.n = G.n
        self.id = get_id()
        self.node_dict = {}
        self.vol = 0

        self.adj_matrix = G.adj_matrix

        for i in range(self.n):
            next(self.id)
        for i in range(self.n):
            x = next(self.id)
            v = 0
            for w in G.adj_table[i].values():
                v += w
            self.node_dict[x] = tree_node(ID=x, vol=v, g=v, partition=[i], adj_table={k + G.n: v for k, v in G.adj_table[i].items()})
            self.vol += v

        self.root = next(self.id)
        self.node_dict[self.root] = tree_node(ID=self.root, vol=self.vol, g=0, children=set(range(self.n, 2 * self.n)), child_cut=self.vol / 2, partition=[i for i in range(self.n)])

        for i in range(self.n, self.n * 2):
            self.node_dict[i].parent = self.root
            self.node_dict[i].d = 1
            self.node_dict[self.root].h = max(self.node_dict[self.root].h, self.node_dict[i].h + 1)

    def print_tree(self):
        edges = []
        q = Queue()
        q.put(self.root)
        id = dict()
        pn = len(self.node_dict)
        while (q.empty() is False):
            a = q.get()
            if (len(self.node_dict[a].children) == 0):
                id[a] = a
                continue
            else:
                pn -= 1
                id[a] = pn
            for b in sorted(self.node_dict[a].children):
                q.put(b)
                assert self.node_dict[a].h == self.node_dict[b].h + 1
                if (len(self.node_dict[b].children) == 0):
                    b = self.node_dict[b].partition[0]
                # print(a, b)
                edges.append([a, b])
        # print('--------------')
        edges = np.array(edges)
        edges = edges.reshape(-1)
        edges = np.array([id[i] for i in edges]).reshape(-1, 2)
        parent = np.array([-1] * len(self.node_dict))
        for a, b in edges:
            parent[b] = a
        # for k, v in id.items():
        #     print('dict', k, v)
        return parent

    def k_HCSE(self, k):
        # global nnn
        # nnn += 1
        # print(nnn)
        sys.setrecursionlimit(max(self.n + 5, 1000))
        # if (k <= 1):
        #     return self.print_tree()
        T = self

        while (T.node_dict[T.root].h < k):
            ok = False
            U = [[] for i in range(T.node_dict[T.root].h)]
            spar = [0 for i in range(T.node_dict[T.root].h)]
            for i in BFS(T):
                if (T.node_dict[i].h == 0):
                    continue
                delta_entory, new_id, new_node_dict = Stretch_Compress(i, T)
                if (True or abs(delta_entory) > 1e-8):
                    ok = True
                    U[T.node_dict[i].d].append((i, new_id, new_node_dict))
                    spar[T.node_dict[i].d] += delta_entory

            for i in range(T.node_dict[T.root].h):
                if (len(U[i]) != 0):
                    spar[i] /= len(U[i])
                else:
                    spar[i] = 1e18
            if (ok is False):
                break
            j = np.argmin(spar)
            delta1 = spar[j] * len(U[j])
            # print(delta1)
            for ori_id, new_id, new_node_dict in U[j]:
                update(ori_id, new_id, T, new_node_dict)
            T.node_dict[T.root].d = 0
            T.node_dict = {i: T.node_dict[i] for i in BFS(T)}
            update_dep_h(T.root, T.node_dict)
            # print_tree(T.root, T.node_dict)

        T.node_dict = {i: T.node_dict[i] for i in BFS(T)}
        update_dep_h(T.root, T.node_dict)
        q = Queue()
        q.put(T.root)
        while (not q.empty()):
            a = q.get()
            T.node_dict[a].adj_table = None
            for b in T.node_dict[a].children:
                q.put(b)
            if (len(T.node_dict[a].children) == 0):
                node = T.node_dict.pop(a)
                id = node.partition[0]
                node.ID = id
                parent = node.parent
                T.node_dict[parent].children.remove(a)
                T.node_dict[parent].children.add(id)
                T.node_dict[id] = node
        return self.print_tree()


def calc_entory(root, node_dict, g_vol):
    entory = 0
    q = Queue()
    q.put(root)
    while (q.empty() is False):
        a = q.get()
        if (node_dict[a].parent is not None and node_dict[a].vol > 0):
            t = -(node_dict[a].g / g_vol) * math.log(node_dict[a].vol / node_dict[node_dict[a].parent].vol, 2)
            entory += t
            node_dict[a].entory = t
        if (node_dict[a].h > 0):
            for b in node_dict[a].children:
                q.put(b)
    return entory


def BFS(T):
    q = Queue()
    q.put(T.root)
    while (not q.empty()):
        a = q.get()
        yield a
        for b in T.node_dict[a].children:
            q.put(b)


def update_dep_h(x, node_dict):
    if (len(node_dict[x].children) == 0):
        node_dict[x].h = 0
        return 0
    node_dict[x].h = 0
    for y in node_dict[x].children:
        if (y not in node_dict):
            continue
        node_dict[y].d = node_dict[x].d + 1
        node_dict[x].h = max(node_dict[x].h, update_dep_h(y, node_dict) + 1)
    return node_dict[x].h


def CompressDelta(node1, p_node):
    a = node1.child_cut
    v1 = node1.vol
    v2 = p_node.vol
    return a * math.log(v2 / v1, 2)


def StretchDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol
    v2 = node2.vol
    g1 = node1.g
    g2 = node2.g
    v12 = v1 + v2
    # return (- 2 * cut_v * math.log(g_vol / v12, 2)) / g_vol
    return ((v1 - g1) * math.log(v12 / v1, 2) + (v2 - g2) * math.log(v12 / v2, 2) - 2 * cut_v * math.log(g_vol / v12, 2)) / g_vol


# def StretchDelta(cut, g_vol, v12, v_p):
#     if (v12 <= 1e-6 or v_p <= 1e-6):
#         return 0
#     # return random.randint(0,10000)
#     return 2 * cut / g_vol * math.log(v12 / v_p, 2)
#
#
# def CompressDelta(cut, g_vol, v12, v_p):
#     if (v12 <= 1e-6 or v_p <= 1e-6):
#         return 0
#     # return random.randint(0,10000)
#     return -2 * cut / g_vol * math.log(v12 / v_p, 2)


def calc_cut(X, Y, adj_matrix):
    cut_v = 0
    for x in X:
        for y in Y:
            cut_v += adj_matrix[x][y]
    return cut_v


def merge(parent_ab, a, b, cut_v, node_dict):
    vol = node_dict[a].vol + node_dict[b].vol
    g = node_dict[a].g + node_dict[b].g - 2 * cut_v
    parent = node_dict[a].parent
    children = {a, b}
    partition = []
    # partition = node_dict[a].partition + node_dict[b].partition
    h = max(node_dict[a].h, node_dict[b].h) + 1
    d = node_dict[a].d
    node_dict[parent_ab] = tree_node(ID=parent_ab, vol=vol, g=g, parent=parent, children=children, partition=partition, h=h, child_cut=cut_v)
    node_dict[a].parent = parent_ab
    node_dict[b].parent = parent_ab
    grand = node_dict[parent_ab].parent
    node_dict[grand].children.remove(a)
    node_dict[grand].children.remove(b)
    node_dict[grand].children.add(parent_ab)
    node_dict[grand].child_cut -= cut_v

    if (len(node_dict[a].adj_table) < len(node_dict[b].adj_table)):
        a, b = b, a
    for k, v in node_dict[a].adj_table.items():
        if node_dict[k].merged is False:
            node_dict[parent_ab].adj_table[k] = v
    for k, v in node_dict[b].adj_table.items():
        if node_dict[k].merged is False:
            node_dict[parent_ab].adj_table[k] = v + node_dict[parent_ab].adj_table.get(k, 0)
    for k, v in node_dict[parent_ab].adj_table.items():
        node_dict[k].adj_table[parent_ab] = v


def compress(x, parent_x, node_dict):
    q = Queue()
    q.put(x)
    while (not q.empty()):
        a = q.get()
        if (node_dict[a].h == 0):
            continue
        for b in node_dict[a].children:
            q.put(b)
        node_dict[a].d -= 1

    node_dict[x].removed = True
    node_dict[parent_x].child_cut += node_dict[x].child_cut
    node_dict[parent_x].children.remove(x)
    for i in node_dict[x].children:
        node_dict[i].parent = parent_x
        node_dict[parent_x].children.add(i)
    while parent_x is not None:
        flag = False
        for i in node_dict[parent_x].children:
            if (node_dict[i].h + 1 == node_dict[parent_x].h):
                flag = True
                break
        if flag == False:
            node_dict[parent_x].h -= 1
            parent_x = node_dict[parent_x].parent
        else:
            break


def is_complete_graph(root, node_list):
    # return False
    n = len(node_list[root].children)
    for i in node_list[root].children:
        if (len(node_list[i].adj_table) != n - 1):
            return False
    return True


def Stretch_tree(root, node_dict, T):
    g_vol = node_dict[root].vol - node_dict[root].g
    stretch_heap = []
    for i in node_dict[root].children:
        for j, w in node_dict[i].adj_table.items():
            if (i < j):
                cut_v = w
                # v12 = node_dict[i].vol + node_dict[j].vol
                # v_p = node_dict[node_dict[i].parent].vol
                delta = StretchDelta(node_dict[i], node_dict[j], cut_v, g_vol)
                heapq.heappush(stretch_heap, (delta, i, j, cut_v))

    unmerged_cnt = len(node_dict[root].children)
    delta_sum = 0
    while (unmerged_cnt > 2 and len(stretch_heap) > 0):
        delta, a, b, cut_v = heapq.heappop(stretch_heap)
        if (abs(delta) < 1e-8):
            continue
        if (node_dict[a].merged or node_dict[b].merged):
            continue
        # print('merge ' + str(node_dict[a].partition) + '   ' + str(node_dict[b].partition) + 'delta = %f' % (delta))
        delta_sum += delta
        node_dict[a].merged = node_dict[b].merged = True
        parent_ab = next(T.id)
        merge(parent_ab, a, b, cut_v, node_dict)
        node_dict[parent_ab].h = 1
        i = parent_ab
        for j, w in node_dict[i].adj_table.items():
            if (node_dict[j].merged == False):
                cut_v = w
                delta = StretchDelta(node_dict[i], node_dict[j], cut_v, g_vol)
                heapq.heappush(stretch_heap, (delta, i, j, cut_v))
        unmerged_cnt -= 1
    update_dep_h(root, node_dict)
    return delta_sum


def Compress_tree(root, node_dict, T, k=2):
    g_vol = node_dict[root]
    delta_sum = 0
    compress_heap = []

    q = Queue()
    q.put(root)
    while (not q.empty()):
        a = q.get()
        for b in node_dict[a].children:
            if (node_dict[b].h != 0):
                cut_v = node_dict[b].child_cut
                delta = CompressDelta(node_dict[b], node_dict[a])
                heapq.heappush(compress_heap, (delta, b, a))
                q.put(b)

    while (len(compress_heap) > 0):
        delta, x, parent_x = heapq.heappop(compress_heap)
        if (x not in node_dict[parent_x].children):
            continue
        if (node_dict[x].removed or node_dict[parent_x].removed):
            continue
        if (node_dict[x].d + node_dict[x].h <= k):
            continue
        delta1 = CompressDelta(node_dict[x], node_dict[parent_x])
        if (abs(delta - delta1) > 1e-8):
            continue
        # print('compress ' + str(x) + '   ' + str(parent_x) + ' delta = %f' % (delta))

        delta_sum += delta
        compress(x, parent_x, node_dict)
        if (parent_x != root):
            delta = CompressDelta(node_dict[parent_x], node_dict[node_dict[parent_x].parent])
            heapq.heappush(compress_heap, (delta, parent_x, node_dict[parent_x].parent))
        for i in node_dict[x].children:
            if (node_dict[i].h > 0):
                delta = CompressDelta(node_dict[i], node_dict[parent_x])
                heapq.heappush(compress_heap, (delta, i, parent_x))
    return delta_sum


def Stretch_Compress(root, T):
    if (len(T.node_dict[root].children) <= 2):
        # return 0, 0, 0
        new_root = next(T.id)
        b = next(T.id)
        new_node_dict = dict()
        node = copy.copy(T.node_dict[root])
        new_node_dict[b] = tree_node(ID=b, vol=node.vol, g=node.g, parent=node.parent, children=node.children, partition=node.partition
                                     , h=0, d=1)
        node.d = 0
        node.h = 1
        node.children = set([b])
        node.ID = new_root
        new_node_dict[new_root] = node
        return 0, new_root, new_node_dict
    if (T.node_dict[root].h == 1):
        if (is_complete_graph(root, T.node_dict)):
            # return 0, 0, 0
            new_root = next(T.id)
            b = next(T.id)
            new_node_dict = dict()
            node = copy.copy(T.node_dict[root])
            new_node_dict[b] = tree_node(ID=b, vol=node.vol, g=node.g, parent=node.parent, children=node.children, partition=node.partition
                                         , h=0, d=1)
            node.d = 0
            node.h = 1
            node.children = set([b])
            node.ID = new_root
            new_node_dict[new_root] = node
            return 0, new_root, new_node_dict

    new_node_dict = {}
    new_root = next(T.id)
    t = T.node_dict[root]
    new_node_dict[new_root] = tree_node(ID=new_root, vol=t.vol, g=t.g, parent=None, partition=t.partition
                                        , adj_table=t.adj_table, h=1, d=0, child_cut=t.child_cut)
    ori_new_dict = dict()
    for i in T.node_dict[root].children:
        id = next(T.id)
        t = T.node_dict[i]
        new_node_dict[id] = tree_node(ID=id, vol=t.vol, g=t.g, parent=new_root, children=t.children, partition=t.partition
                                      , h=0, d=1, child_cut=t.child_cut)
        new_node_dict[new_root].children.add(id)
        ori_new_dict[i] = id
    edges = []
    for i in T.node_dict[root].children:
        a = ori_new_dict[i]
        for b, w in T.node_dict[i].adj_table.items():
            if (b in ori_new_dict):
                b = ori_new_dict[b]
                new_node_dict[a].adj_table[b] = w
                if (a < b):
                    edges.append((a, b, w))

    ori_entory = calc_entory(new_root, new_node_dict, T.vol)

    delta_sum = 0
    delta_sum += Stretch_tree(new_root, new_node_dict, T)
    ret = False
    if ret:
        return delta_sum, new_root, new_node_dict
    delta_sum += Compress_tree(new_root, new_node_dict, T)

    for i in copy.copy(new_node_dict[new_root].children):
        if (new_node_dict[i].h == 0):
            id = next(T.id)
            node = new_node_dict[i]
            new_node_dict[id] = tree_node(ID=id, vol=node.vol, g=node.g, parent=new_root, children=set([node.ID]), partition=node.partition
                                          , h=1, d=1)
            node.h = 0
            node.d = 2
            node.parent = id
            new_node_dict[new_root].children.remove(node.ID)
            new_node_dict[new_root].children.add(id)

    new_node_dict = {k: v for k, v in new_node_dict.items() if v.removed is False}
    for i in new_node_dict[new_root].children:
        new_node_dict[i].adj_table.clear()
    for i in ori_new_dict.values():
        new_node_dict[i].adj_table.clear()

    for a, b, w in edges:
        fa = new_node_dict[a].parent
        fb = new_node_dict[b].parent
        if (fa != fb):
            # print(fa, fb)
            t = new_node_dict[fa].adj_table[fb] = new_node_dict[fa].adj_table.get(fb, 0) + w
            new_node_dict[fb].adj_table[fa] = t
        else:
            t = new_node_dict[a].adj_table[b] = new_node_dict[a].adj_table.get(b, 0) + w
            new_node_dict[b].adj_table[a] = t

    update_dep_h(new_root, new_node_dict)
    new_entory = calc_entory(new_root, new_node_dict, T.vol)
    delta_sum = new_entory - ori_entory
    return delta_sum, new_root, new_node_dict


def update(ori_id, new_id, T, new_node_dict):
    for id, node in new_node_dict.items():
        T.node_dict[id] = node
    new_node_dict[new_id].parent = T.node_dict[ori_id].parent
    T.node_dict[ori_id] = new_node_dict[new_id]
    T.node_dict[ori_id].ID = ori_id
    for id, node in new_node_dict.items():
        if (node.d == 1):
            node.parent = ori_id
        if (node.h == 0):
            for child in node.children:
                T.node_dict[child].parent = id


def main():
    adj_matrix = [
        0, 0, 0, 0, 0, 0, 0, 0, 8,
        0, 0, 3, 0, 0, 10, 0, 0, 0,
        0, 3, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 2, 0, 4, 0, 0, 0, 0,
        0, 0, 0, 4, 0, 4, 1, 0, 0,
        0, 10, 0, 0, 4, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 5, 3,
        0, 0, 0, 0, 0, 0, 5, 0, 4,
        8, 0, 0, 0, 0, 0, 3, 4, 0,
    ]
    adj_matrix = np.array(adj_matrix).reshape(9, 9)
    # G = Graph(adj_matrix=adj_matrix)
    G = Graph(path='d:\\test.in')
    T = Tree(G)
    print(calc_entory(T.root, T.node_dict, T.vol))
    parent = T.k_HCSE(3)
    for i, fa, in enumerate(parent):
        if (fa != -1):
            print(fa, i)
    print(calc_entory(T.root, T.node_dict, T.vol))
    # update_node(T.node_dict)


# def update_depth(tree):
#     # set leaf depth
#     wait_update = [k for k, v in tree.items() if v.children is None]
#     while wait_update:
#         for nid in wait_update:
#             node = tree[nid]
#             if node.children is None:
#                 node.h = 0
#             else:
#                 node.h = tree[list(node.children)[0]].h + 1
#         wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])
#
#
# def update_node(tree):
#     update_depth(tree)
#     d_id = [(v.h, v.ID) for k, v in tree.items()]
#     d_id.sort()
#     new_tree = {}
#     for k, v in tree.items():
#         print(k, v)
#         n = copy.deepcopy(v)
#         n.ID = d_id.index((n.h, n.ID))
#         if n.parent is not None:
#             n.parent = d_id.index((n.h + 1, n.parent))
#         if n.children is not None:
#             n.children = [d_id.index((n.h - 1, c)) for c in n.children]
#         n = n.__dict__
#         n['depth'] = n['h']
#         new_tree[n['ID']] = n
#     return new_tree


if __name__ == '__main__':
    import time

    a = time.time()
    main()
    b = time.time()
    print("cost time = %f" % (b - a))
