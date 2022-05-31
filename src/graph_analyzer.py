from collections import defaultdict
import tvm
from tvm import relay
import network
import relay_util


class BufferInfo:
    def __init__(self, device_type):
        self.device_type = device_type
        self.size = 0
        self.firstUse = -1
        self.lastUse = -1
        self.static = False
        self.label = ""


def get_post_dfs_order_exprs(expr):
    out = []

    def visit(node):
        out.append(node)

    relay.analysis.post_order_visit(expr, visit)
    return out


def getExprLabel(expr):
    if isinstance(expr, relay.Call):
        if isinstance(expr.op, tvm.ir.op.Op):
            return expr.op.name
        elif isinstance(expr.op, relay.Function):
            opNames = []
            next = expr.op.body
            while isinstance(next, relay.Call):
                if isinstance(next.op, tvm.ir.op.Op):
                    opNames.append(relay_util.abbreviateOpName(next.op.name))
                else:
                    break
                next = relay_util.getCallInput(next)
            return "_".join(reversed(opNames))
    return ""


class GraphAnalyzer(relay.ExprVisitor):
    def __init__(self, explicitInputs=None):
        super().__init__()
        self.exprToBufInfos = {}
        self.exprToLabel = {}
        self.explicitInputs = explicitInputs

    def run(self, func):
        # This is no longer available in python. Maybe the DeviceAwareExprVisitor will become available?
        # self.nodeDeviceMap = relay.analysis.collect_device_info(func)
        self.nodeDeviceMap = {}
        self.orderedExprs = get_post_dfs_order_exprs(func)

        # Create bufs for model inputs.
        for param in func.params:
            isStatic = False
            if self.explicitInputs != None:
                if param.name_hint not in self.explicitInputs:
                    isStatic = True
            self.updateOrCreateBufInfos(param, static=isStatic)

        # Create bufs for intermediates and outputs.
        self.visit(func.body)

        # Verify consistency.
        for bufInfos in self.exprToBufInfos.values():
            firstUse = -2
            lastUse = -2
            for bufInfo in bufInfos:
                if firstUse == -2:
                    firstUse = bufInfo.firstUse
                    lastUse = bufInfo.lastUse
                else:
                    assert firstUse == bufInfo.firstUse
                    assert lastUse == bufInfo.lastUse

        # Generate labels.
        callLabelId = 0
        varLabelId = 0
        tgiLabelId = 0
        tupLabelId = 0
        for expr in self.orderedExprs:
            if expr in self.exprToBufInfos:
                if isinstance(expr, relay.Call):
                    self.exprToLabel[expr] = "C" + str(callLabelId)
                    extraLabel = self.exprToBufInfos[expr][0].label
                    if extraLabel != "":
                        self.exprToLabel[expr] += "_" + extraLabel
                    callLabelId += 1
                elif isinstance(expr, relay.Var):
                    self.exprToLabel[expr] = "V" + str(varLabelId)
                    varLabelId += 1
                elif isinstance(expr, relay.TupleGetItem):
                    self.exprToLabel[expr] = "TGI" + str(tgiLabelId)
                    tgiLabelId += 1
                elif isinstance(expr, relay.Tuple):
                    self.exprToLabel[expr] = "TUP" + str(tupLabelId)
                    tupLabelId += 1

    # Ensures updated BufferInfo for expr. For call arguments, pass argOfCall=callExpr.
    def updateOrCreateBufInfos(self, expr, argOfCall=None, static=False):
        if expr not in self.exprToBufInfos:
            bufInfos = self.createBufInfos(expr, static)
        else:
            bufInfos = self.exprToBufInfos[expr]

        useNode = argOfCall if argOfCall != None else expr
        t = self.orderedExprs.index(useNode)
        for bufInfo in bufInfos:
            if argOfCall != None:
                bufInfo.lastUse = max(bufInfo.lastUse, t)
            else:
                if bufInfo.firstUse == -1:
                    bufInfo.firstUse = t
                else:
                    bufInfo.firstUse = min(bufInfo.firstUse, t)

    # Create new BufferInfo for expr.
    def createBufInfos(self, expr, static):
        assert expr not in self.exprToBufInfos
        bufInfos = []
        device_type = self.nodeDeviceMap.get(expr, 0)

        def makeBuf(t):
            bufInfo = BufferInfo(device_type)
            bufInfo.size = relay_util.RelayType(t).getSize()
            bufInfo.static = static
            bufInfo.label = getExprLabel(expr)
            return bufInfo

        if isinstance(expr.checked_type, relay.ty.TupleType):
            for t in expr.checked_type.fields:
                bufInfos.append(makeBuf(t))
        else:
            bufInfos.append(makeBuf(expr.checked_type))

        self.exprToBufInfos[expr] = bufInfos
        return bufInfos

    def visit_function(self, func):
        # Do not recurse into sub functions.
        pass

    def visit_constant(self, const):
        self.updateOrCreateBufInfos(const, static=True)

    def visit_call(self, call):
        # Buffer used as output.
        self.updateOrCreateBufInfos(call)

        # Buffers used as input.
        for arg in call.args:
            self.visit(arg)
            self.updateOrCreateBufInfos(arg, argOfCall=call)

    def visit_tuple_getitem(self, t):
        self.updateOrCreateBufInfos(t)
        self.updateOrCreateBufInfos(t.tuple_value)
        super().visit_tuple_getitem(t)

    def makeNet(self):
        n = network.Network()

        exprToBufs = defaultdict(list)
        self.bufToExpr = {}
        self.bufToBufInfo = {}
        for expr, bufInfos in self.exprToBufInfos.items():
            for bufNum, bufInfo in enumerate(bufInfos):
                if expr in self.exprToLabel:
                    labelExt = "_out" + (str(bufNum) if bufNum != 0 else "")
                    buf = n.addBuf(self.exprToLabel[expr] + labelExt, bufInfo.size, bufInfo.static)
                    exprToBufs[expr].append(buf)
                    self.bufToExpr[buf] = expr
                    self.bufToBufInfo[buf] = bufInfo

        self.exprToOp = {}
        for expr, bufsInfo in self.exprToBufInfos.items():
            if isinstance(expr, relay.Call):
                inBufs = []
                for arg in expr.args:
                    inBufs.extend(exprToBufs[arg])
                op = n.addOp(self.exprToLabel[expr], inBufs, exprToBufs[expr])
                self.exprToOp[expr] = op
            elif isinstance(expr, relay.TupleGetItem):
                inBuf = exprToBufs[expr.tuple_value][expr.index]
                op = n.addOp(self.exprToLabel[expr], [inBuf], exprToBufs[expr])
            elif isinstance(expr, relay.Tuple):
                inBufs = []
                for field in expr.fields:
                    inBufs.extend(exprToBufs[field])
                op = n.addOp(self.exprToLabel[expr], inBufs, exprToBufs[expr])

        n.createGraph()
        return n

    def getSched(self):
        sched = network.Schedule()
        for expr in self.orderedExprs:
            if expr in self.exprToOp:
                sched.addOp(self.exprToOp[expr])
        return sched

    def getExprFromBuf(self, buf):
        return self.bufToExpr[buf]

    def getBufInfoFromBuf(self, buf):
        return self.bufToBufInfo[buf]

    def getStaticBufInfos(self):
        out = []
        for expr, bufInfos in self.exprToBufInfos.items():
            for bufInfo in bufInfos:
                if bufInfo.static:
                    out.append((expr, bufInfo))
        return out
