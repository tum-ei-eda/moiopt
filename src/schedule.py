import ilpsolver
import networkx as nx


class SchedOpt:
    def __init__(self):
        self.ilp = ilpsolver.ILPSolver()
        self.s = self.ilp.s

    def solve(self, G):
        # Insert single source and sink nodes.
        sources = [n for n, deg in G.in_degree() if deg == 0]
        sinks = [n for n, deg in G.out_degree() if deg == 0]
        sourceNode = "^"
        sinkNode = "$"
        G = nx.DiGraph(G)
        if len(sources) == 0 or len(sinks) == 0:
            assert len(sources) == 0 and len(sinks) == 0
            sources = [sinkNode]
        G.add_edges_from([(sourceNode, s) for s in sources])
        G.add_edges_from([(s, sinkNode) for s in sinks])
        G.nodes[sourceNode]["outsize"] = 1
        G.nodes[sinkNode]["outsize"] = 1

        # Solve for each independent problem.
        cutG = nx.Graph(G)
        br = list(nx.bridges(cutG))
        cutG.remove_edges_from(br)
        scheds = {}
        for comp in nx.connected_components(cutG):
            subG = G.subgraph(comp).copy()
            if len(subG.nodes) == 1:
                n = list(subG.nodes)[0]
                scheds[n] = [n]
            else:
                sched = self.solveImpl(subG)
                scheds[sched[0]] = sched

        # Reassemble in original order and remove source/sink.
        sched = []
        for n in nx.topological_sort(G):
            sched.extend(scheds.get(n, []))
        return sched[1:-1]

    def solveImpl(self, G):
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
            for succ in G.successors(node):
                self.ilp.lessThan(t[i], t[nodeToIdx[succ]])
            # t[i] != t[j]  forall  j=0...i-1
            for j in range(0, i):
                self.ilp.notEqual(t[i], t[j])

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

        # for x in range(numNodes):
        #     activeNodes = ""
        #     for i in range(numNodes):
        #         val = int(h[x][i].solution_value())
        #         print(val, end="")
        #         if val == 1:
        #             activeNodes += idxToNode[i] + " "
        #     print("", int(m[x].solution_value()), activeNodes)
        # print("max mem:", int(maxM.solution_value()))

        sched = []
        for i, node in sorted(nodeOrder, key=lambda x: x[0]):
            sched.append(node)
        return sched


def solve(G):
    return SchedOpt().solve(G)
