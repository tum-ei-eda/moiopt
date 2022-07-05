import exec_timeout
import ilpsolver
import schedule_sp_dec
import schedule_sp
import networkx as nx


def is_sp_graph_impl(esg):
    G = esg.view()
    if len(G.nodes) < 3:
        return True
    spType, esg1, esg2 = schedule_sp_dec.decompose_sp_graph(esg)
    if spType == schedule_sp_dec.SPType.INVALID:
        return False
    G1 = esg1.view()
    G2 = esg2.view()
    isSP = True
    if len(G1.edges) != 1:
        isSP &= is_sp_graph_impl(esg1)
    if len(G2.edges) != 1:
        isSP &= is_sp_graph_impl(esg2)
    return isSP


def is_sp_graph(G):
    return is_sp_graph_impl(schedule_sp_dec.EfficientSubgraph(G))


class DummyNode:
    def __init__(self, parent=None):
        self.parent = parent

    def __repr__(self):
        return "Dummy" + (("(" + str(self.parent) + ")") if self.parent else str(id(self)))

    def __lt__(self, other):
        return id(self) < id(other)


# Convert ML model graph to task graph with weights on nodes (computation cost) and edges (communication cost).
def makeTaskGraph(G):
    TG = nx.DiGraph(G)
    for sink in [n for n in TG.nodes if TG.out_degree(n) == 0]:
        TG.add_edge(sink, DummyNode())
    for n in TG.nodes:
        for e in TG.out_edges(n):
            TG.edges[e]["w"] = TG.nodes[n]["outsize"]
        TG.nodes[n]["w"] = 0
    return TG


# Convert task graph to cumulative weight model.
def makeCumGraph(G):
    G = makeTaskGraph(G)
    CG = nx.DiGraph()
    endN = {}
    for n in G.nodes:
        endN[n] = DummyNode(n)
        CG.add_edge(n, endN[n])
        w = G.nodes[n]["w"]
        CG.nodes[n]["cw"] = w
        CG.nodes[endN[n]]["cw"] = -w
    for e in G.edges:
        CG.add_edge(endN[e[0]], e[1])
        w = G.edges[e]["w"]
        CG.nodes[e[0]]["cw"] += w
        CG.nodes[endN[e[1]]]["cw"] -= w
    return CG


class SchedOpt:
    def __init__(self):
        self.ilp = ilpsolver.ILPSolver()
        self.s = self.ilp.s

    def solve(self, G):
        # Insert single source and sink nodes.
        sources = [n for n, deg in G.in_degree() if deg == 0]
        sinks = [n for n, deg in G.out_degree() if deg == 0]
        sourceNode = DummyNode()
        sinkNode = DummyNode()
        G = nx.DiGraph(G)
        if len(sources) == 0 or len(sinks) == 0:
            assert len(sources) == 0 and len(sinks) == 0
            sources = [sinkNode]
        G.add_edges_from([(sourceNode, s) for s in sources])
        G.add_edges_from([(s, sinkNode) for s in sinks])
        G.nodes[sourceNode]["outsize"] = 1
        G.nodes[sinkNode]["outsize"] = 1

        sched = self.solveSubproblem(schedule_sp_dec.EfficientSubgraph(G))

        # Remove any dummy nodes.
        return [n for n in sched if not isinstance(n, DummyNode)]

    def solveSubproblem(self, esg):
        G = esg.view()
        if len(G.edges) == 1:
            e = list(G.edges)[0]
            return (e[0], e[1])
        spType, esg1, esg2 = schedule_sp_dec.decompose_sp_graph(esg)
        # Try to split into independent problems by considering series composition.
        if spType == schedule_sp_dec.SPType.SERIES:
            G1 = esg1.view()
            G2 = esg2.view()
            if len(G1.nodes) == 1:
                sched1 = list(G1.nodes)
            else:
                sched1 = self.solveSubproblem(esg1)
            if len(G2.nodes) == 1:
                sched2 = list(G2.nodes)
            else:
                sched2 = self.solveSubproblem(esg2)
            return sched1 + sched2[1:]

        if is_sp_graph_impl(esg):
            # Solve series-parallel graphs with polynomial time algorithm.
            CG = makeCumGraph(G)
            return self.solveWithTimeout(CG, lambda g: schedule_sp.sp_schedule(g)[0])

        # Solve general graphs with MILP.
        return self.solveWithTimeout(G, lambda g: self.solveILP(g))

    def solveWithTimeout(self, G, solveFunc, timeout=0.5):
        def inExternalProcess():
            nodesToIds = {node: i for i, node in enumerate(G.nodes)}
            sched = solveFunc(G)
            return [nodesToIds[node] for node in sched]

        idsToNodes = {i: node for i, node in enumerate(G.nodes)}
        try:
            ids = exec_timeout.exec_timeout(timeout, inExternalProcess)
            return tuple([idsToNodes[id] for id in ids])
        except TimeoutError:
            return tuple(nx.topological_sort(G))

    def solveILP(self, G):
        numNodes = len(G.nodes)

        # 0 <= t[i] < N
        t = []
        w = []
        idxToNode = {}
        nodeToIdx = {}
        for i, node in enumerate(G.nodes):
            t.append(self.s.IntVar(0, numNodes - 1, "t_" + str(i)))
            w.append(G.nodes[node]["outsize"])
            idxToNode[i] = node
            nodeToIdx[node] = i

        for i, node in enumerate(G.nodes):
            # t[i] > t_preds
            for pred in G.predecessors(node):
                self.ilp.greaterThan(t[i], t[nodeToIdx[pred]])
            # t[i] < t_succs  /  Optional?
            # for succ in G.successors(node):
            #    self.ilp.lessThan(t[i], t[nodeToIdx[succ]])
            # t[i] != t[j]  forall  j=0...i-1
            # for j in range(0, i):
            #    self.ilp.notEqual(t[i], t[j])

        h = []
        for x in range(numNodes):
            h.append([])
            for i in range(numNodes):
                h[x].append(self.s.BoolVar("h_" + str(x) + "_" + str(i)))

                # The currently scheduled node must be live.
                # t[i] == x  =>  h[x][i] = 1
                # t[i] <= x-1 or -t[i] <= -x-1 or -h[x][i] <= -1
                cts = self.ilp.makeOrConstraintMulti(3, [x - 1, -x - 1, -1])
                cts[0].SetCoefficient(t[i], 1)
                cts[1].SetCoefficient(t[i], -1)
                cts[2].SetCoefficient(h[x][i], -1)

                # Helper variable is false if any successor of node is yet to be scheduled.
                # t[k] >= x  =>  z = 0
                # t[k] <= x-1 or z <= 0
                z = self.s.BoolVar("z_" + str(x) + "_" + str(i))
                succs = list(G.successors(idxToNode[i]))
                for succ in succs:
                    k = nodeToIdx[succ]
                    ct1, ct2 = self.ilp.makeOrConstraint(x - 1, 0)
                    ct1.SetCoefficient(t[k], 1)
                    ct2.SetCoefficient(z, 1)

                # Node i is live if it was scheduled in the past and any successor is yet to be scheduled.
                # t[i] < x and (1-z)  =>  h[x][i] = 1
                # -t[i] <= -x or -z <= -1 or -h[x][i] <= -1
                cts = self.ilp.makeOrConstraintMulti(3, [-x, -1, -1])
                cts[0].SetCoefficient(t[i], -1)
                cts[1].SetCoefficient(z, -1)
                cts[2].SetCoefficient(h[x][i], -1)

        # m[x] = sum_i(h[x][i] * W[i])
        m = []
        for x in range(numNodes):
            m.append(self.s.IntVar(0, self.s.Infinity(), "m_" + str(x)))
            ct = self.s.Constraint(0, 0, self.ilp.nextId("ct_eqsum"))
            ct.SetCoefficient(m[x], -1)
            for i in range(numNodes):
                ct.SetCoefficient(h[x][i], w[i])

        # minimize: max(m[x])
        maxM = self.s.IntVar(0, self.s.Infinity(), "maxM")
        for x in range(numNodes):
            self.ilp.greaterEqual(maxM, m[x])
        obj = self.ilp.getObjectiveFunc()
        obj.SetCoefficient(maxM, 1)

        self.ilp.solve()

        nodeOrder = []
        for i in range(numNodes):
            nodeOrder.append((int(t[i].solution_value()), idxToNode[i]))

        sched = []
        for i, node in sorted(nodeOrder, key=lambda x: x[0]):
            sched.append(node)
        return tuple(sched)


def solve(G):
    return SchedOpt().solve(G)
