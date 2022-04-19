import networkx as nx
import schedule


def escapeDotName(name):
    return name.replace(" ", "_").replace(".", "_")


# Represents a buffer that is used in LayerOps.
class Buffer:
    def __init__(self, name, size, static=False):
        self.name = name
        self.size = size
        self.static = static

    def isStatic(self):
        return self.static

    def plot(self):
        return (
            escapeDotName(self.name)
            + '[label="'
            + self.name
            + " ("
            + str(self.size)
            + ')", style=filled, color='
            + ("gray40" if self.static else "gray80")
            + "];\n"
        )

    def __repr__(self):
        return f"Buffer('{self.name}', {self.size}, {self.static})"


# Represents an abstract operation that takes inputs and produces outputs.
class LayerOp:
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    def getInputs(self):
        return self.inputs

    def getOutputs(self):
        return self.outputs

    def getDynInputs(self):
        return [buf for buf in self.inputs if not buf.isStatic()]

    def getInSize(self):
        return sum([buf.size for buf in self.getDynInputs()])

    def getOutSize(self):
        return sum([buf.size for buf in self.outputs])

    # Returns the required memory size for non-static inputs and outputs.
    def getSize(self):
        return self.getInSize() + self.getOutSize()

    def plot(self):
        ownName = escapeDotName(self.name)
        out = ownName + "[shape=box, style=filled, color=coral];\n"
        for i in self.inputs:
            out += i.plot() + escapeDotName(i.name) + " -> " + ownName + ";\n"
        for i in self.outputs:
            out += i.plot() + ownName + " -> " + escapeDotName(i.name) + ";\n"
        return out

    def __repr__(self):
        inBufs = ",".join([str(inBuf) for inBuf in self.inputs])
        outBufs = ",".join([str(outBuf) for outBuf in self.outputs])
        return f"LayerOp('{self.name}')[{inBufs} -> {outBufs}]"


# Represents a collection of Buffers and LayerOps to form a data flow graph.
class Network:
    def __init__(self):
        self.bufs = []
        self.ops = []

    def addBuf(self, *args):
        buf = Buffer(*args)
        self.bufs.append(buf)
        return buf

    def addOp(self, *args):
        op = LayerOp(*args)
        self.ops.append(op)
        return op

    def createGraph(self):
        self.g = nx.DiGraph()

        if len(self.ops) == 1:
            self.g.add_node(self.ops[0])
        else:
            for op in self.ops:
                for outBuf in op.getOutputs():
                    for opTarget in self.ops:
                        if outBuf in opTarget.getInputs():
                            self.g.add_edge(op, opTarget)

        for op in self.g.nodes:
            self.g.nodes[op]["outsize"] = op.getOutSize()

    def createSchedules(self):
        scheds = []
        for ops in nx.all_topological_sorts(self.g):
            sched = Schedule()
            for op in ops:
                sched.addOp(op)
            scheds.append(sched)

        return scheds

    def createBestSchedule(self):
        try:
            sched = Schedule()
            for op in schedule.solve(self.g):
                sched.addOp(op)
            return sched
        except RuntimeError:
            return self.createAnySchedule()

    def createAnySchedule(self):
        sched = Schedule()
        for op in nx.topological_sort(self.g):
            sched.addOp(op)
        return sched

    def getInOps(self):
        return [op for op, deg in self.g.in_degree() if deg == 0]

    def getOutOps(self):
        return [op for op, deg in self.g.out_degree() if deg == 0]

    def getInBufs(self):
        return [buf for op in self.getInOps() for buf in op.getDynInputs()]

    def getOutBufs(self):
        return [buf for op in self.getOutOps() for buf in op.getOutputs()]

    def plot(self):
        out = ""
        for op in self.ops:
            out += op.plot()
        return out

    def __repr__(self):
        out = "Network([\n"
        for op in self.ops:
            out += "  " + str(op) + "\n"
        return out + "])"


# Represents a linear order of LayerOps.
class Schedule:
    def __init__(self):
        self.sched = []

    def addOp(self, op):
        self.sched.append(op)

    def plot(self):
        out = ""
        prevOp = None
        for op in self.sched:
            out += op.plot()
            if prevOp:
                out += escapeDotName(prevOp.name) + " -> " + escapeDotName(op.name) + " [color=red];\n"
            prevOp = op
        return out

    def __repr__(self):
        return "[" + "] > [".join([op.name for op in self.sched]) + "]"
