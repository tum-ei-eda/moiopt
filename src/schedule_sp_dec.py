import networkx as nx
from enum import Enum
from collections import defaultdict


SPType = Enum("SPType", "INVALID SERIES PARALLEL")


class EfficientSubgraph:
    def __init__(self, G, filterN=None, filterE=None):
        self.G = G
        self.fN = filterN if filterN != None else []
        self.fE = filterE if filterE != None else []

    def view(self):
        return nx.restricted_view(self.G, self.fN, self.fE)

    def subgraph(self, filterN=None, filterE=None):
        filterN = self.fN + filterN if filterN != None else self.fN
        if filterE != None and filterE != []:
            G = self.view()
            nodeToReducedDeg = defaultdict(int)
            for e in filterE:
                nodeToReducedDeg[e[0]] += 1
                nodeToReducedDeg[e[1]] += 1
            for n, redDeg in nodeToReducedDeg.items():
                if G.degree(n) <= redDeg and n not in filterN:
                    filterN.append(n)
        filterE = self.fE + filterE if filterE != None else self.fE
        return EfficientSubgraph(self.G, filterN, filterE)

    def subgraph_keep(self, keepN=None, keepE=None):
        G = self.view()
        filterN = [n for n in G.nodes if n not in keepN] if keepN != None else []
        filterE = [e for e in G.edges if e not in keepE] if keepE != None else []
        return self.subgraph(filterN, filterE)


# Decomposes the given graph into two series or parallel graphs.
# Not the most efficient, but simple to understand. An improvement would be:
# "The Recognition of Series Parallel Digraphs" by J. Valdes et al.
def decompose_sp_graph(esg):
    G = esg.view()
    assert len(G.edges) > 1

    sources = []
    sinks = []
    for n in G.nodes:
        if G.in_degree(n) == 0:
            sources.append(n)
        if G.out_degree(n) == 0:
            sinks.append(n)
    if len(sources) != 1 or len(sinks) != 1:
        return SPType.INVALID, None, None

    source = sources[0]
    sink = sinks[0]
    if G.out_degree(source) == 1:
        return SPType.SERIES, esg.subgraph_keep(keepE=list(G.out_edges(source))), esg.subgraph([source])
    elif G.in_degree(sink) == 1:
        return SPType.SERIES, esg.subgraph([sink]), esg.subgraph_keep(keepE=list(G.in_edges(sink)))

    # Check for other series composition by removing nodes and checking if we get two connected components.
    for n in G.nodes:
        GR = nx.Graph(G)
        GR.remove_node(n)
        comps = list(nx.connected_components(GR))
        if len(comps) == 2:
            # Make sure G1 is connected to the source.
            if source in comps[1]:
                assert sink in comps[0]
                comps[0], comps[1] = comps[1], comps[0]
            filterN1 = []
            filterN2 = []
            for n in G.nodes:
                if n in comps[0]:
                    filterN2.append(n)
                elif n in comps[1]:
                    filterN1.append(n)
            return SPType.SERIES, esg.subgraph(filterN1), esg.subgraph(filterN2)
        assert len(comps) == 1

    # Check for parallel composition by picking a source edge and walking its connections.
    # First, the special case where source and sink are directly connected must be considered.
    outEdges = list(G.out_edges(source))
    if (source, sink) in outEdges:
        filterN2 = []
        for n in G.nodes:
            if n not in [source, sink]:
                filterN2.append(n)
        return SPType.PARALLEL, esg.subgraph(filterE=[(source, sink)]), esg.subgraph(filterN2)
    startNode = outEdges[0][1]
    visitedNodes = {source, sink, startNode}
    currentNodes = {startNode}
    while len(currentNodes) != 0:
        nextNodes = set()
        for n in currentNodes:
            nextNodes.update([e[1] for e in G.out_edges(n)])
            nextNodes.update([e[0] for e in G.in_edges(n)])
        for n in visitedNodes:
            nextNodes.discard(n)
        visitedNodes |= nextNodes
        currentNodes = nextNodes
    # If we reached all nodes, there must be some cross-connection, so there is no parallel composition.
    if len(G.nodes) == len(visitedNodes):
        return SPType.INVALID, None, None
    # Otherwise, there is a parallel composition.
    filterN1 = []
    filterN2 = []
    for n in G.nodes:
        if n in [source, sink]:
            continue
        if n in visitedNodes:
            filterN1.append(n)
        else:
            filterN2.append(n)
    return SPType.PARALLEL, esg.subgraph(filterN1), esg.subgraph(filterN2)
