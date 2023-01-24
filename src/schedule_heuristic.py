from collections import defaultdict
import networkx as nx


# Finds a list of (startNode, endNode, list of paths) in the graph G, where each element
# represents a parallel subgraph from start to end with no path nodes having branches.
def findInnerParallelPaths(G):
    allPaths = []
    for n in G.nodes:
        # Find splitting nodes.
        if G.out_degree(n) > 1:
            targetToPaths = defaultdict(list)
            for e in G.out_edges(n):
                path = [e]

                # Follow straight path.
                while G.in_degree(e[1]) == 1 and G.out_degree(e[1]) == 1:
                    e = list(G.out_edges(e[1]))[0]
                    path.append(e)

                targetToPaths[e[1]].append(path)

            for t, paths in targetToPaths.items():
                # Only store parallel paths.
                if len(paths) > 1:
                    # Store start, end and all paths between them.
                    allPaths.append((n, t, paths))

    return allPaths


# Replaces allPaths subgraphs with a straight connection from their start to end nodes.
def reduceGraph(G, allPaths):
    for start, end, paths in allPaths:
        # Remove intermediate nodes.
        for p in paths:
            for e in p[:-1]:
                G.remove_node(e[1])
        # Add direct edge.
        G.add_edge(start, end)
    return G


# Like findParallelPaths, but also finds nested parallel paths.
def findParallelPaths(G):
    G = nx.DiGraph(G)

    out = []
    while True:
        allPaths = findInnerParallelPaths(G)
        if len(allPaths) == 0:
            break
        out.extend(allPaths)
        G = reduceGraph(G, allPaths)

    return out


# Finds schedule for DFG G that has minimal memory footprint.
def schedule(G):
    pp = findParallelPaths(G)

    w = {}
    for n in G.nodes:
        w[n] = G.nodes[n]["outsize"]

    subScheds = defaultdict(list)
    for start, end, ps in pp:
        pathHVIs = {}
        for p in ps:
            if len(p) == 1:
                continue
            maxNodeIndex = None
            maxNodeW = -1
            for i, e in enumerate(p[:-1]):
                if w[e[1]] > maxNodeW:
                    maxNodeW = w[e[1]]
                    maxNodeIndex = i
            minNodeIndex = None
            minNodeW = maxNodeW
            for i, e in enumerate(p[maxNodeIndex + 1:-1]):
                if w[e[1]] < minNodeW:
                    minNodeW = minNodeW
                    minNodeIndex = i + maxNodeIndex + 1
            if minNodeIndex == None:
                pathIndex = maxNodeIndex
            else:
                pathIndex = minNodeIndex
            p = tuple(p)
            pathHVIs[p] = (maxNodeW - minNodeW, pathIndex)

        subSched = []
        sortedHVIs = {k: v for k, v in sorted(pathHVIs.items(), key=lambda x: x[1][0], reverse=True)}
        for p, hvi in sortedHVIs.items():
            i = hvi[1] + 1
            subSched.extend([e[1] for e in p[:i]])
        for p, hvi in sortedHVIs.items():
            i = hvi[1] + 1
            subSched.extend([e[1] for e in p[i:-1]])
        w[start] = next(iter(sortedHVIs.items()))[1][0]
        subScheds[start].append(subSched)

    sources = [n for n, deg in G.in_degree() if deg == 0]
    sinks = [n for n, deg in G.out_degree() if deg == 0]
    assert len(sources) == 1 and len(sinks) == 1

    # Flatten sub scheds.
    def flatten(s):
        out = [s[0]]
        for subSched in s[1]:
            for n in subSched:
                if n in subScheds:
                    out.extend(flatten((n, subScheds[n])))
                else:
                    out.append(n)
        return out
    flatSubScheds = flatten(next(reversed(subScheds.items())))

    n = sources[0]
    outSched = []
    toVisit = []
    while True:
        outSched.append(n)
        if n == sinks[0]:
            break

        toVisit.extend([e[1] for e in G.out_edges(n)])
        toVisit = list(dict.fromkeys(toVisit))
        if len(toVisit) == 1:
            n = toVisit[0]
            toVisit = []
            continue

        # Pick the out edges that is scheduled soonest.
        minIndex = 9e99
        nextN = None
        for v in toVisit:
            i = flatSubScheds.index(v) if v in flatSubScheds else 9e90
            if i < minIndex:
                minIndex = i
                nextN = v
        assert nextN != None
        n = nextN
        toVisit.remove(n)

    return tuple(outSched)
