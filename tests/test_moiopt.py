import testutil as tu
import tvm


e = tu.relayOp("100", None)
e = tu.relayOp("dense10", e)
e = tu.relayOp("add", e)
mod = tvm.IRModule.from_expr(e)
c_mod, c_params = tu.buildTVM(mod, {}, "tmp.so")

class ModelInfo:
    pass
class TensorInfo:
    pass
mInfo = ModelInfo()
mInfo.inTensors = [TensorInfo()]
mInfo.inTensors[0].name = "inp"
mInfo.inTensors[0].shape = (1, 100)
mInfo.inTensors[0].ty = "float32"
mInfo.outTensors = [TensorInfo()]
mInfo.outTensors[0].shape = (1, 10)
mInfo.outTensors[0].ty = "float32"
out = tu.runTVM("tmp.so", c_params, mInfo)
print("out:", out)
