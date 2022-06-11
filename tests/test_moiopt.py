import testutil as tu
import graph_analyzer
import memplanner
import moiopt
import numpy as np
import tvm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Input,
    Concatenate,
    ReLU,
    DepthwiseConv2D,
)


def memSize(mod):
    analyzer = graph_analyzer.GraphAnalyzer()
    analyzer.run(mod["main"])
    n = analyzer.makeNet()
    sched = n.createBestSchedule()
    planner = memplanner.MemoryPlanner(sched)

    def cbLayout(layout):
        # drawLayout("lay" + str(i), layout)
        # writePlot("plt" + str(i), n.plot())
        # plot.drawLayoutChart("lc" + str(i), layout, planner)
        return {n.bufs[0]: layout.getSize()}

    sz = memplanner.memLayoutWithTimeout(n, planner, cbLayout)
    return sz.popitem()[1]


testId = 0


def test(model, expectedSizeFactor):
    global testId
    print("=====", testId, expectedSizeFactor, "=====")
    testId += 1
    mod, params, modelInfo = tu.kerasToRelay(model)

    # @tvm.register_func("tvm.relay.plan_memory")
    # def _plan_memory(func):
    #    return plan_memory.plan_memory(func)

    @tvm.register_func("relay.backend.PostPass")
    def _post_pass(mod, params):
        # print(mod)
        preSz = memSize(mod)
        mod = moiopt.MOIOPTPass()(mod)
        postSz = memSize(mod)
        print("sz", preSz, "->", postSz, "is:", postSz / preSz)
        tu.leq(postSz, preSz * expectedSizeFactor)
        # print(mod)
        return mod

    c_mod, c_params = tu.buildTVM(mod, params, "tmp.so")

    tvm._ffi.registry.remove_global_func("relay.backend.PostPass")
    c_mod, c_paramsref = tu.buildTVM(mod, params, "tmpref.so")

    # if os.path.getsize("tmp.so") > os.path.getsize("tmpref.so"):
    #    print("Test failed: The pass increased the model size")
    #    ctx.recordFail()

    out = tu.runTVM("tmp.so", c_params, modelInfo)
    outref = tu.runTVM("tmpref.so", c_paramsref, modelInfo, "last")

    assert len(out) == len(outref)
    for i, t in enumerate(out):
        refT = outref[i]
        diff = np.abs(t - refT)
        if t.dtype == "float32":
            # A small error is unavoidable with floating point if the order of operations is not the same.
            okay = np.all(np.less(diff, 0.000001))
        else:
            okay = np.all(diff == 0)
        tu.ass(okay, "Data mismatch:\n" + str(t) + "\n" + str(refT))


test(tu.kerasDense([5], 1, 1), 1)
test(tu.kerasDense([200], 100, 100), 0.8)
test(tu.kerasDense([19391], 4), 0.11)
test(tu.kerasDense([20], 4), 0.5)
test(tu.kerasDense([5, 20], 3), 0.5)
test(tu.kerasDense([10, 1], 3), 0.7)
test(tu.kerasDense([5, 20, 5], 3), 0.7)
test(tu.kerasDense([40, 20], 10), 0.8)
test(tu.kerasConv(10, 1), 0.4)
test(tu.kerasConv(11, 1), 0.4)
test(tu.kerasConv(10, 1, padding="same"), 0.4)
test(tu.kerasConv(10, 3), 0.4)
test(tu.kerasConv(10, 3, padding="same"), 0.4)

inp = Input(shape=(10,))
tmp = Dense(20)(inp)
tmp2 = Dense(40)(inp)
tmp = Dense(10)(tmp)
tmp2 = Dense(20)(tmp2)
tmp = Concatenate()([tmp, tmp2])
tmp = Dense(1)(tmp)
test(Model(inputs=inp, outputs=tmp), 0.9)

inp = Input(shape=(10,))
splt = Dense(100)(inp)
tmp = Dense(1000)(splt)
tmp2 = Dense(1000)(splt)
tmp = Dense(100)(tmp)
tmp2 = Dense(100)(tmp2)
tmp = Concatenate()([tmp, tmp2])
test(Model(inputs=inp, outputs=tmp), 0.97)

inp = Input(shape=(32, 32, 3))
tmp = Conv2D(filters=32, kernel_size=(3, 3))(inp)
tmp = MaxPooling2D()(tmp)
tmp = Conv2D(filters=64, kernel_size=(3, 3))(tmp)
tmp = MaxPooling2D()(tmp)
tmp = Flatten()(tmp)
tmp = Dense(10)(tmp)
test(Model(inputs=inp, outputs=tmp), 0.35)

inp = Input(shape=(32, 32, 3))
tmp = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(inp)
tmp = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(tmp)
test(Model(inputs=inp, outputs=tmp), 0.4)

inp = Input(shape=(32, 32, 3))
tmp = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(inp)
tmp = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding="same")(tmp)
test(Model(inputs=inp, outputs=tmp), 0.45)

inp = Input(shape=(151, 151, 64))
tmp = ReLU()(inp)
tmp = MaxPooling2D((3, 3), strides=(2, 2))(tmp)
test(Model(inputs=inp, outputs=tmp), 0.7)

# inp = Input(shape=(151, 151, 3))
# tmp = Conv2D(filters=64, kernel_size=(1, 1))(inp)
# tmp = MaxPooling2D((3, 3), strides=(2, 2))(tmp)
# test(Model(inputs=inp, outputs=tmp), 1)  # some issue with this one?

inp = Input(shape=(16, 16, 8))
tmp = ReLU()(inp)
tmp = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(tmp)
tmp = MaxPooling2D(pool_size=(2, 2), strides=(4, 4))(tmp)
tmp = ReLU()(tmp)
test(Model(inputs=inp, outputs=tmp), 0.45)
