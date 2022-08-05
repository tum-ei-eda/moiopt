from tvm import relay
import relay_util


class ReshapeInfo:
    def __init__(self, call, paramIndex, parentFuncParam, reshapeFuncParam, newShape):
        self.call = call
        self.paramIndex = paramIndex
        self.parentFuncParam = parentFuncParam
        self.reshapeFuncParam = reshapeFuncParam
        self.newShape = newShape


# Merges reshape only nodes into the following function.
@relay.transform.function_pass(opt_level=0)
class FixReshapesPass(relay.ExprMutator):
    def __init__(self):
        super().__init__()
        self.reshapeOnlyCalls = []
        self.reshapeInfos = []

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, relay.Function):
            attrName = "relay.reshape_only"
            if attrName in call.op.attrs:
                if call.op.attrs[attrName] == 1:
                    # TODO: handle functions that contains more than one reshape.
                    if isinstance(call.op.body.args[0], relay.Var):
                        # TODO: could handle dyn.reshape with more effort.
                        if call.op.body.op.name == "reshape":
                            self.reshapeOnlyCalls.append(call)

        newArgs = []
        for arg in call.args:
            newArgs.append(self.visit(arg))
            if arg in self.reshapeOnlyCalls:
                self.reshapeOnlyCalls.append(newArgs[-1])

        if isinstance(call.op, relay.Function):
            argsFromReshapes = []
            for i, arg in enumerate(newArgs):
                if arg in self.reshapeOnlyCalls:
                    argsFromReshapes.append((i, arg))

            if len(argsFromReshapes) > 0:
                newFnParams = {}
                for i, arg in argsFromReshapes:
                    origParam = arg.op.params[0]
                    newFnParams[i] = relay.var(origParam.name_hint, origParam.type_annotation)

                prevReshapeInfos = self.reshapeInfos
                self.reshapeInfos = []
                for i, arg in argsFromReshapes:
                    newShape = arg.op.body.attrs.newshape
                    self.reshapeInfos.append(ReshapeInfo(call, i, call.op.params[i], newFnParams[i], newShape))
                fnBody = self.visit(call.op.body)
                fnParams = [self.visit(p) for p in call.op.params]
                self.reshapeInfos = prevReshapeInfos

                for i, arg in argsFromReshapes:
                    fnParams[i] = newFnParams[i]
                    newArgs[i] = arg.args[0]
                newFn = relay.function.FunctionWithFields(call.op, fnParams, fnBody)
                return relay.Call(newFn, newArgs, call.attrs, call.type_args, call.span)

        return relay.Call(self.visit(call.op), newArgs, call.attrs, call.type_args, call.span)

    def visit_var(self, var):
        for info in self.reshapeInfos:
            if var == info.parentFuncParam:
                return relay.op.reshape(info.reshapeFuncParam, newshape=info.newShape)

        return var


class MACEstimator(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.adds = 0

    def getCount(self):
        return self.count

    def visit_call(self, call):
        if not isinstance(call.op, relay.Function):
            if call.op.name == "nn.conv2d" or call.op.name == "qnn.conv2d":
                outShape = relay_util.getShape(call)
                wShape = relay_util.getShape(call.args[1])
                wSize = 1
                for dim in wShape:
                    wSize *= dim
                self.count += wSize * outShape[1] * outShape[2]
            elif call.op.name == "qnn.dense" or call.op.name == "nn.dense" or call.op.name == "nn.contrib_dense_pack":
                wShape = relay_util.getShape(call.args[1])
                wSize = 1
                for dim in wShape:
                    wSize *= dim
                self.count += wSize
            elif call.op.name == "add":
                outShape = relay_util.getShape(call)
                outSize = 1
                for dim in outShape:
                    outSize *= dim
                self.adds += outSize
        super().visit_call(call)
