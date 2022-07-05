import re
import functools

import tvm
from tvm import relay


reEndsInNum = re.compile("\\d+$")


def divRoundUp(sz, word_sz):
    return (sz + word_sz - 1) // word_sz


# Replaces the given matchCall with a call provided by insertFunc.
# insertFunc takes the mutated original call as input and should return the replacement for it.
@relay.transform.function_pass(opt_level=0)
class ReplaceCallPass(relay.ExprMutator):
    def __init__(self, matchCall, insertFunc):
        super().__init__()
        self.matchCall = matchCall
        self.insertFunc = insertFunc

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        newCall = super().visit_call(call)
        if call == self.matchCall:
            return self.insertFunc(newCall)
        return newCall


# Replaces matchCall with the return value of insertFunc and remembers rememberCall until after type inference.
class ReplaceAndRemember(relay.ExprMutator):
    def __init__(self, matchCall, insertFunc, rememberCall=None):
        super().__init__()
        self.matchCall = matchCall
        self.insertFunc = insertFunc
        self.rememberCall = rememberCall
        self.rememberedCall = None

    def run(self, mod):
        newFunc = self.visit(mod["main"])
        mod = tvm.IRModule.from_expr(newFunc)
        posVisitor = GraphPositionVisitor()
        callPos = posVisitor.getPos(mod, self.rememberedCall)
        mod = relay.transform.InferType()(mod)
        self.rememberedCall = posVisitor.getCall(mod, callPos)
        return mod

    def visit_call(self, call):
        newCall = super().visit_call(call)
        if call == self.matchCall:
            newCall = self.insertFunc(newCall)
        if call == self.rememberCall:
            self.rememberedCall = newCall
        return newCall


# Returns mappings between calls and graph positions which are stable between InferType passes.
class GraphPositionVisitor(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.pos = 0
        self.callToFind = None
        self.posToFind = None

    def getPos(self, mod, call):
        self.callToFind = call
        self.visit(mod["main"])
        self.reset()
        return self.foundPos

    def getCall(self, mod, pos):
        self.posToFind = pos
        self.visit(mod["main"])
        self.reset()
        return self.foundCall

    def visit_call(self, call):
        if call == self.callToFind:
            self.foundPos = self.pos
        if self.pos == self.posToFind:
            self.foundCall = call
        self.pos += 1
        super().visit_call(call)


# Wrap expressions in this class to compare them between different modules.
class ExprCmpHelper:
    def __init__(self, e):
        self.e = e
        self.eTy = self.getTextType()

    def getTextType(self):
        if isinstance(self.e, relay.Call):
            return "c"
        elif isinstance(self.e, relay.Constant):
            return "C"
        elif isinstance(self.e, relay.Var):
            return "V"
        elif isinstance(self.e, relay.Tuple):
            return "t"
        elif isinstance(self.e, relay.TupleGetItem):
            return "T"
        else:
            return "u"

    def __eq__(self, other):
        if self.eTy != other.eTy:
            return False
        if self.eTy == "C":
            # The passes do not create new data objects, so we can just check for object id equality.
            return self.e.data.same_as(other.e.data)
        elif self.eTy == "V":
            return self.e.name_hint == other.e.name_hint
        elif self.eTy == "c":
            if len(self.e.args) != len(other.e.args):
                return False
            for i in range(0, len(self.e.args)):
                if ExprCmpHelper(self.e.args[i]) != ExprCmpHelper(other.e.args[i]):
                    return False
        return True


# Searches mod for equivalent expression of exprFromOtherMod.
def findFromOtherModule(mod, exprFromOtherMod):
    otherExprHelper = ExprCmpHelper(exprFromOtherMod)

    def efficientCompare(e):
        return ExprCmpHelper(e) == otherExprHelper

    return FindCall().run(mod, efficientCompare)[0]


# Finds an expr in mod by matching to the given pred.
class FindCall(relay.ExprVisitor):
    def run(self, mod, pred):
        self.pred = pred
        self.results = []
        self.visit(mod["main"])
        return self.results

    def visit_call(self, call):
        if self.pred(call):
            self.results.append(call)
        return super().visit_call(call)


# Returns the type of an expression that has not been inferred yet. Costly!
@functools.lru_cache
def getCheckedType(expr):
    tmpMod = tvm.IRModule.from_expr(expr)
    tmpMod = relay.transform.InferType()(tmpMod)
    return tmpMod["main"].body.checked_type


# Helper to get information about a type in relay.
class RelayType:
    def __init__(self, arg):
        if isinstance(arg, relay.Constant):
            self.dtype = arg.data.dtype
            self.shape = arg.data.shape
        else:
            if isinstance(arg, relay.Expr):
                if arg._checked_type_ == None:
                    arg = getCheckedType(arg)
                else:
                    arg = arg.checked_type
            assert isinstance(arg, relay.ty.TensorType)  # TODO tuples
            self.dtype = arg.dtype
            self.shape = arg.shape

        match = reEndsInNum.search(self.dtype)
        assert match != None
        self.tySz = int(match[0])

    def getShape(self):
        # Translate from dynamic to static types.
        return tuple(int(dim) for dim in self.shape)

    def isFloat(self):
        return self.dtype.startswith("float")

    def isInt(self):
        return self.dtype.startswith("int")

    def getDType(self):
        return self.dtype

    def getTypeSizeBits(self):
        return self.tySz

    def getTypeSize(self):
        return divRoundUp(self.tySz, 8)

    def getSize(self):
        size = 1
        for dim in self.getShape():
            size *= dim
        size *= self.getTypeSize()
        return int(size)


def getShape(expr):
    return RelayType(expr).getShape()


def getSize(expr):
    return RelayType(expr).getSize()


def getTypeSize(expr):
    return RelayType(expr).getTypeSize()


def getDType(expr):
    return RelayType(expr).getDType()


def isDepthwiseConv(call):
    if call.attrs.groups == 1:
        # Quick exit for performance.
        return False
    inShape = getShape(call.args[0])
    inLayout = call.attrs.data_layout
    kernelShape = call.args[1].data.shape
    kernelLayout = call.attrs.kernel_layout
    groups = call.attrs.groups
    return relay.op.strategy.is_depthwise_conv2d(inShape, inLayout, kernelShape, kernelLayout, groups)


def isFlatten(call):
    inShape = getShape(call.args[0])
    outShape = getShape(call)
    if inShape == outShape:
        return False
    return len(outShape) == 1 or outShape.count(1) > (len(outShape) - 2)


def hasOverlappingInput(call):
    if call.op.name == "nn.conv2d":
        kSize = call.attrs["kernel_size"]
        strides = call.attrs["strides"]
        if kSize[0] > strides[0] or kSize[1] > strides[1]:
            return True
    elif call.op.name in ["max_pool2d", "avg_pool2d"]:
        poolSize = call.attrs["pool_size"]
        strides = call.attrs["strides"]
        if poolSize[0] > strides[0] or poolSize[1] > strides[1]:
            return True
    elif call.op.name == "nn.contrib_dense_pack":
        return True
    return False


def getCallInput(call):
    if isinstance(call.args[0], relay.Constant):
        if call.op.name in ["add", "take", "multiply"]:
            return call.args[1]
        else:
            raise RuntimeError(f"unexpected call input arg for {call.op.name}")
    else:
        return call.args[0]


def abbreviateOpName(name):
    n = {
        "nn.contrib_dense_pack": "dense",
        "fixed_point_multiply": "mult",
        "subtract": "sub",
        "nn.max_pool2d": "maxpool",
        "right_shift": "rshft",
        "nn.conv2d": "conv",
    }
    return n.get(name, name)


# Converts an expr to a string that is useful for debugging and not too large.
def exprToStr(expr):
    out = ""
    if isinstance(expr, relay.Call):
        out += "call_"
        if isinstance(expr.op, tvm.ir.op.Op):
            out += expr.op.name
        elif isinstance(expr.op, relay.Function):
            opNames = []
            next = expr.op.body
            while isinstance(next, relay.Call):
                if isinstance(next.op, tvm.ir.op.Op):
                    opNames.append(abbreviateOpName(next.op.name))
                else:
                    break
                next = getCallInput(next)
            out += "_".join(reversed(opNames))
    elif isinstance(expr, relay.Var):
        out += "var_" + expr.name_hint
    else:
        out += "unknown_expr"
    return out + "[" + str(getShape(expr)) + "]"


def normalizePadding(padding):
    if isinstance(padding, int) or isinstance(padding, tvm.tir.IntImm):
        return [int(padding)] * 4
    elif len(padding) == 4:
        return [int(p) for p in padding]
    elif len(padding) == 2:
        return [int(p) for p in padding] * 2
    elif len(padding) == 1:
        return [int(padding[0])] * 4
    else:
        raise RuntimeError("unexpected padding format")


def getNormalizedPaddingValue(padding, getWidthValue, getBeginValue):
    if getWidthValue:
        return padding[1] if getBeginValue else padding[3]
    else:
        return padding[0] if getBeginValue else padding[2]


def getNormalizedPaddingPair(padding, getWidthValue):
    if getWidthValue:
        return (padding[1], padding[3])
    else:
        return (padding[0], padding[2])


def getPaddingValue(padding, getWidthValue, getBeginValue):
    padding = normalizePadding(padding)
    return getNormalizedPaddingValue(padding, getWidthValue, getBeginValue)
