import functools
import networkx as nx
import schedule_sp_dec


# This implements the following works:
# "Scheduling Series-Parallel Task Graphs to Minimize Peak Memory" by E. Kayaaslan et al.
# "An Application of Generalized Tree Pebbling to Sparse Matrix Factorization" by J. W. H. Liu


# Pebble cost of a node.
def cost(T, node):
    return T.nodes[node]["w"]


# Accumulated pebble cost of a sequence until a node on a tree.
@functools.cache
def accumulated_cost(T, node, sequence):
    prevIndex = sequence.index(node) - 1
    if prevIndex >= 0:
        c = accumulated_cost(T, sequence[prevIndex], sequence)
    else:
        c = 0
    c += cost(T, node)
    for n in T.predecessors(node):
        c -= cost(T, n)
    return c


# Maximum accumulated pebble cost of a sequence on a tree.
def peak_cost(T, sequence):
    return max([accumulated_cost(T, n, sequence) for n in sequence])


# Pebble cost sequence.
def pcost(T, sequence):
    h = []
    v = [-1]
    H = []
    V = []
    accWithIndex = [(0, 0)] + [(accumulated_cost(T, n, sequence), i) for i, n in enumerate(sequence)]
    while v[-1] != len(sequence) - 1:
        # Largest sequence index that maximizes the accumulated cost.
        hill, h_index = max(accWithIndex[v[-1] + 1 :])
        h.append(h_index)
        H.append(hill)
        # Largest sequence index that minimizes the accumulated cost.
        valley, v_index = min(accWithIndex[h[-1] + 1 :], key=lambda x: (x[0], -x[1]))
        v.append(v_index)
        V.append(valley)
    v = v[1:]
    return list(zip(H, V)), list(zip(h, v))


# Combine sequences of subtrees for optimal ordering.
def combine(T, root, sequences):
    segments = []
    for sequence in sequences:
        # Determine valley segments.
        HV, hv = pcost(T, sequence)
        for i in range(len(HV)):
            segValue = HV[i][0] - HV[i][1]
            segBeginIndex = hv[i - 1][1] + 1 if i > 0 else 0
            seg = [sequence[idx] for idx in range(segBeginIndex, hv[i][1] + 1)]
            segments.append((segValue, seg))
    # Arrange segments by their segment value.
    ordering = []
    for seg in sorted(segments, key=lambda x: x[0], reverse=True):
        ordering.extend(seg[1])
    ordering.append(root)
    return tuple(ordering)


# Returns the optimal ordering of the given tree.
def pebble_ordering(T, root):
    children = list(T.predecessors(root))
    if len(children) == 0:
        return tuple([root])
    orderings = tuple(pebble_ordering(T, node) for node in children)
    return combine(T, root, orderings)


def cumulative_weight(T):
    return sum([T.nodes[n]["cw"] for n in T.nodes])


def tree_schedule(T, root):
    TwithoutRoot = nx.DiGraph(T)
    TwithoutRoot.remove_node(root)
    for n in T.nodes:
        if n == root:
            T.nodes[n]["w"] = cumulative_weight(TwithoutRoot)
        else:
            subNodes = nx.ancestors(T, n) | {n}
            subT = nx.DiGraph(T)
            for n2 in T.nodes:
                if n2 not in subNodes:
                    subT.remove_node(n2)
            T.nodes[n]["w"] = cumulative_weight(subT)
    return pebble_ordering(T, root)


def reverse_graph(G):
    RG = nx.reverse(G)
    for n in G.nodes:
        RG.nodes[n]["cw"] = -G.nodes[n]["cw"]
    return RG


def pc_schedule(G, S, T, source, sink):
    GS = nx.DiGraph(G)
    GT = nx.DiGraph(G)
    for n in G.nodes:
        if n not in S:
            GS.remove_node(n)
        if n not in T:
            GT.remove_node(n)
    s1 = tree_schedule(reverse_graph(GS), source)
    s2 = tree_schedule(GT, sink)
    s1 = tuple(reversed(s1))
    return s1 + s2


def adopt_weights(toG, fromG):
    for n in toG.nodes:
        toG.nodes[n]["cw"] = fromG.nodes[n]["cw"]


def linearize_graph(G, sched):
    LG = nx.DiGraph()
    for i in range(1, len(sched)):
        LG.add_edge(sched[i - 1], sched[i])
    adopt_weights(LG, G)
    return LG


# Returns an optimal schedule of the given series-parallel graph, along with its min-w-cut.
def sp_schedule_impl(esg):
    G = esg.view()

    # Base case where there is only a single edge.
    if len(G.edges) == 1:
        e = list(G.edges)[0]
        return (e[0], e[1]), {e[0]}, {e[1]}

    # Decompose G into G1 and G2 which are either in series or parallel to each other.
    spType, esg1, esg2 = schedule_sp_dec.decompose_sp_graph(esg)
    assert spType != schedule_sp_dec.SPType.INVALID

    sched1, S1, T1 = sp_schedule_impl(esg1)
    sched2, S2, T2 = sp_schedule_impl(esg2)

    G1 = esg1.view()
    G2 = esg2.view()

    # Series composition
    if spType == schedule_sp_dec.SPType.SERIES:
        # Let S, T be one of the topo cuts with minimum cutwidth
        T1 = T1.union(set(G2.nodes))
        S2 = S2.union(set(G1.nodes))
        cutWidth1 = sum([G.nodes[n]["cw"] for n in S1])
        cutWidth2 = sum([G.nodes[n]["cw"] for n in S2])
        if cutWidth1 < cutWidth2:
            S = S1
            T = T1
        else:
            S = S2
            T = T2
        # Remove duplicate connecting node.
        sched = sched1 + sched2[1:]
        return sched, S, T

    # Parallel composition
    else:
        G1 = linearize_graph(G1, sched1)
        G2 = linearize_graph(G2, sched2)
        S = S1.union(S2)
        T = T1.union(T2)
        G = nx.compose(G1, G2)
        sched = pc_schedule(G, S, T, sched1[0], sched2[-1])
        return sched, S, T


def sp_schedule(G):
    accumulated_cost.cache_clear()
    return sp_schedule_impl(schedule_sp_dec.EfficientSubgraph(G))
