import networkx as nx
from enum import Enum


SPType = Enum("SPType", "INVALID SERIES PARALLEL")


# Decomposes the given graph into two series or parallel graphs.
# Not the most efficient, but simple to understand. An improvement would be:
# "The Recognition of Series Parallel Digraphs" by J. Valdes et al.
def decompose_sp_graph(G):
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
        G1 = G.edge_subgraph(G.out_edges(source)).copy()
        G2 = nx.DiGraph(G)
        G2.remove_node(source)
        return SPType.SERIES, G1, G2
    elif G.in_degree(sink) == 1:
        G1 = nx.DiGraph(G)
        G1.remove_node(sink)
        G2 = G.edge_subgraph(G.in_edges(sink)).copy()
        return SPType.SERIES, G1, G2

    # Check for other series composition by removing nodes and checking if we get two connected components.
    for n in G.nodes:
        GR = nx.Graph(G)
        GR.remove_node(n)
        comps = list(nx.connected_components(GR))
        if len(comps) == 2:
            G1 = nx.DiGraph(G)
            G2 = nx.DiGraph(G)
            # Make sure G1 is connected to the source.
            if source in comps[1]:
                assert sink in comps[0]
                comps[0], comps[1] = comps[1], comps[0]
            for n in G.nodes:
                if n in comps[0]:
                    G2.remove_node(n)
                elif n in comps[1]:
                    G1.remove_node(n)
            return SPType.SERIES, G1, G2
        assert len(comps) == 1

    # Check for parallel composition by picking a source edge and walking its connections.
    # First, the special case where source and sink are directly connected must be considered.
    outEdges = list(G.out_edges(source))
    if (source, sink) in outEdges:
        G1 = nx.DiGraph(G)
        G2 = nx.DiGraph(G)
        G1.remove_edge(source, sink)
        for n in G.nodes:
            if n not in [source, sink]:
                G2.remove_node(n)
        return SPType.PARALLEL, G1, G2
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
    G1 = nx.DiGraph(G)
    G2 = nx.DiGraph(G)
    for n in G.nodes:
        if n in [source, sink]:
            continue
        if n in visitedNodes:
            G1.remove_node(n)
        else:
            G2.remove_node(n)
    return SPType.PARALLEL, G1, G2
