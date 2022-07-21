import sys
import random
import os
import itertools
import unittest
import atexit

sys.path.append("../src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import networkx as nx

import network
import relay_util
import graph_analyzer
import pathdiscovery as pd
from load_tflite_model import load_tflite_model

import tvm
from tvm import relay


__anyFail = False
__tc = unittest.TestCase()
__tc.failureException
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def exit():
    atexit.unregister(exit)
    if __anyFail:
        print("Test failure!")
        sys.exit(1)
    else:
        print("All tests OK!")
        sys.exit(0)


atexit.register(exit)


def __handle_uncaught(t, v, tb):
    fail("Uncaught exception!")
    sys.__excepthook__(t, v, tb)


sys.excepthook = __handle_uncaught

ass = __tc.assertTrue
eq = __tc.assertEqual
neq = __tc.assertNotEqual
geq = __tc.assertGreaterEqual
greater = __tc.assertGreater
leq = __tc.assertLessEqual
less = __tc.assertLess
inst = __tc.assertIsInstance
raises = __tc.assertRaises
almosteq = __tc.assertAlmostEqual


def fail(msg):
    global __anyFail
    __anyFail = True
    print(msg)


# Creates an nx graph from strings where each character represents a node and each string a chain
# connected by directed edges. The last argument is a list of node output sizes in their order of
# appearance. e.g.: makeGraph("abc", "aBc", [1, 2, 3, 4]) creates a simple diamond with source
# a(w=1), going to b(w=2) and B(w=4), both ending in sink c(w=3). Alternatively the weights can be
# passed as a dictionary that maps node names to weight values.
def makeGraph(*args, weights={}):
    G = nx.DiGraph()
    for edgelist in args:
        if len(edgelist) == 1:
            G.add_node(edgelist)
        else:
            G.add_edges_from(nx.utils.pairwise(edgelist))

    if isinstance(weights, list):
        count = 0
        for arg in args:
            for n in arg:
                if "outsize" not in G.nodes[n]:
                    if count >= len(weights):
                        raise RuntimeError("Not enough weights for given graph")
                    G.nodes[n]["outsize"] = weights[count]
                    count += 1
    else:
        for n in G.nodes:
            G.nodes[n]["outsize"] = weights.get(n, 100)

    return G


def makeNetFromGraph(G):
    net = network.Network()

    bufid = 0
    processedNodes = {}

    def addBuf(node):
        nonlocal bufid
        if node not in processedNodes:
            buf = net.addBuf("Buf" + str(bufid), G.nodes[node]["outsize"], False)
            bufid += 1
            processedNodes[node] = buf
        return processedNodes[node]

    nid = 0
    for n in G.nodes:
        inBufs = []
        for inE in G.in_edges(n):
            inBufs.append(addBuf(inE[0]))
        if G.in_degree(n) != 0:
            net.addOp("Op" + str(nid), inBufs, [addBuf(n)])
            nid += 1

    net.createGraph()
    return net


def makeNet(*args):
    G = makeGraph(*args[:-1], weights=args[-1])
    return makeNetFromGraph(G)


def buildTVM(mod, params, exportPath="/tmp/tvmtmp.so"):
    target = tvm.target.Target("c --runtime=c -model=unknown --system-lib")
    cfg = {"tir.disable_vectorize": True}
    with tvm.transform.PassContext(opt_level=3, config=cfg):
        c_mod = relay.build(mod, target, params=params)
        c_mod.export_library(exportPath)
        c_params = c_mod.get_params()
    return c_mod, c_params


lastInputData = None


def runTVM(c_mod, c_params, modelInfo, inputData="random"):
    global lastInputData

    cpuctx = tvm.cpu()
    rtmod = tvm.runtime.load_module(c_mod)
    gmod = tvm.contrib.graph_executor.GraphModule(rtmod["default"](cpuctx))
    gmod.set_input(**c_params)
    currentInputData = {}
    for t in modelInfo.inTensors:
        if isinstance(inputData, np.ndarray):
            data = inputData
        elif isinstance(inputData, dict):
            data = inputData[t.name]
        elif inputData == "random":
            data = np.random.uniform(-1, 1, size=t.shape).astype(t.ty)
        elif inputData == "last":
            assert lastInputData != None
            data = lastInputData[t.name]
        else:
            raise RuntimeError("invalid input data")
        gmod.set_input(t.name, data)
        currentInputData[t.name] = data
    lastInputData = currentInputData
    gmod.run()
    outTensors = []
    for i, t in enumerate(modelInfo.outTensors):
        out = gmod.get_output(i, tvm.nd.empty(t.shape, dtype=t.ty)).asnumpy()
        outTensors.append(out)
    return outTensors


def kerasToRelay(keras_model):
    keras_model.compile()
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    return load_tflite_model(tflite_model)


def kerasDense(layerSizes, inputSize, outputSize=1, activation="tanh"):
    layers = [kl.Input(shape=(inputSize,))]
    layers += [kl.Dense(sz, activation=activation, bias_initializer="uniform") for sz in layerSizes]
    layers += [kl.Dense(outputSize, activation=activation)]
    return km.Sequential(layers)


def kerasConv(nFeatures, kernelSize, activation="tanh", padding="valid", input_shape=(100, 100, 3)):
    return km.Sequential(
        [
            kl.Conv2D(nFeatures, kernelSize, activation=activation, padding=padding, input_shape=input_shape),
            kl.MaxPooling2D(),
            kl.Conv2D(nFeatures, kernelSize, activation=activation, padding=padding),
            kl.MaxPooling2D(),
        ]
    )


def getSchedMemUsage(sched, G):
    keepAlive = {}
    liveSize = 0
    peakMem = 0
    for op in sched:
        liveSize += G.nodes[op]["outsize"]
        peakMem = max(peakMem, liveSize)
        keepAlive[op] = G.out_degree(op)
        for e in G.in_edges(op):
            keepAlive[e[0]] -= 1
            if keepAlive[e[0]] == 0:
                liveSize -= G.nodes[e[0]]["outsize"]
    return peakMem


def verifySched(sched, G):
    if not sched:
        return
    memFound = getSchedMemUsage(sched, G)
    minTested = 9e99
    minSched = None
    allSched = []
    for s in nx.all_topological_sorts(G):
        memTest = getSchedMemUsage(s, G)
        allSched.append((s, memTest))
        if memTest < minTested:
            minTested = memTest
            minSched = s
    if minTested < memFound:
        fail(
            "Test failed: A better schedule was found: "
            + "".join(minSched)
            + " ("
            + str(minTested)
            + ") - Solution: "
            + "".join(sched)
            + " ("
            + str(memFound)
            + ")"
        )


def translateSplitTypes(str):
    import pathdiscovery as pd

    splitTypeLUT = {
        "<": pd.SplitType.LOP,
        "=": pd.SplitType.PARTITIONED,
        ">": pd.SplitType.LIP,
        "~": pd.SplitType.PARTIAL,
        "#": pd.SplitType.FTP,
    }
    assert set(str) <= set(splitTypeLUT.keys())
    return [splitTypeLUT[t] for t in str]


def getNum(s):
    intStr = "".join(itertools.takewhile(str.isdigit, s))
    if intStr == "":
        return None
    else:
        return int(intStr)


def getArg(s, name):
    idx = s.find(name)
    if idx == -1:
        return None
    num = getNum(s[idx + len(name) :])
    if num == None:
        return True
    return num


def relayOp(s, prevExpr):
    if s[0].isdigit():
        if s.endswith("i8"):
            dtype = "int8"
            s = s[:-2]
        else:
            dtype = "float32"
        shape = [1] + [int(dim) for dim in s.split("x")]
        return relay.var("inp", shape=shape, dtype=dtype)

    if not isinstance(prevExpr, list):
        ty = relay_util.RelayType(prevExpr)
        prevShape = ty.getShape()
        dtype = ty.getDType()
    if s.startswith("dense"):
        inSize = prevShape[1]
        outSize = getNum(s[5:])
        if outSize == None:
            outSize = inSize
        bSize = getArg(s[5:], "b")
        if bSize == None:
            bSize = 1
        outType = ""
        if dtype == "int8":
            outType = "int32"
        return relay.nn.contrib_dense_pack(
            prevExpr, relay.const(np.zeros((outSize // bSize, inSize, bSize)), dtype), out_dtype=outType
        )
    elif s.startswith("nndense"):
        inSize = prevShape[1]
        outSize = getNum(s[7:])
        if outSize == None:
            outSize = inSize
        return relay.nn.dense(prevExpr, relay.const(np.zeros((outSize, inSize)), dtype), units=outSize)
    elif s == "add":
        return relay.add(prevExpr, relay.const(np.zeros(prevShape), dtype))
    elif s == "mult":
        if isinstance(prevExpr, list):
            return relay.multiply(prevExpr[0], prevExpr[1])
        return relay.multiply(prevExpr, relay.const(1.0, dtype))
    elif s == "relu":
        return relay.nn.relu(prevExpr)
    elif s.startswith("conv"):
        outSize = getNum(s[4:])
        inSize = prevShape[3]
        if outSize == None:
            outSize = inSize
        kSize = getArg(s[4:], "k")
        if kSize == None:
            kSize = 3
        depthWise = getArg(s[4:], "dw")
        if depthWise != None:
            groups = inSize
            outSize = inSize
            inSize = 1
        else:
            groups = 1
        pad = getArg(s[4:], "pad")
        if pad != None:
            padding = [1, 1, 1, 1]
        else:
            padding = [0, 0, 0, 0]
        stride = getArg(s[4:], "stride")
        if stride == None:
            stride = 1
        outType = ""
        if dtype == "int8":
            outType = "int32"
        return relay.nn.conv2d(
            prevExpr,
            relay.const(np.zeros((outSize, inSize, kSize, kSize), dtype)),
            strides=(stride, stride),
            padding=padding,
            kernel_size=(kSize, kSize),
            data_layout="NHWC",
            groups=groups,
            out_dtype=outType,
        )
    elif s.startswith("pool"):
        size = getNum(s[4:])
        if size == None:
            size = 2
        stride = getArg(s[4:], "stride")
        if stride == None:
            stride = size
        return relay.nn.max_pool2d(prevExpr, pool_size=(size, size), strides=(stride, stride), layout="NHWC")
    elif s == "concat":
        return relay.concatenate([prevExpr], 0)
    elif s == "flatten":
        return relay.reshape(prevExpr, (1, -1))
    elif s == "reshape":
        return relay.reshape(prevExpr, prevShape)
    elif s == "reshapestrip":
        return relay.reshape(prevExpr, prevShape[1:])
    elif s == "reshapeflip":
        newShape = list(prevShape)
        newShape[-1], newShape[-2] = newShape[-2], newShape[-1]
        return relay.reshape(prevExpr, newShape)
    elif s.startswith("cast"):
        size = getNum(s[4:])
        isFloat = getArg(s[4:], "f")
        if size == 8:
            dtype = "int8"
        elif size == 16:
            dtype = "int16"
        elif size == 32:
            dtype = "float32" if isFloat else "int32"
        elif size == 64:
            dtype = "float64" if isFloat else "int64"
        return relay.cast(prevExpr, dtype)
    elif s == "pad":
        return relay.nn.pad(prevExpr, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))
    elif s == "pad1":
        return relay.nn.pad(prevExpr, pad_width=((0, 0), (0, 1), (0, 1), (0, 0)))
    elif s == "pad2":
        return relay.nn.pad(prevExpr, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))
    elif s.startswith("cat"):
        axis = getNum(s[3:])
        if axis == None:
            axis = 0
        return relay.concatenate(prevExpr, axis=axis)
    elif s == "transpose":
        return relay.transpose(prevExpr)
    else:
        raise RuntimeError("not implemented:", s)


def verifyPath(path, expectedSplitTypes, expectedNumPart):
    eq(len(path), len(expectedSplitTypes))

    for i, cfg in enumerate(path):
        eq(cfg.splitType, expectedSplitTypes[i])

        expectedIn = 1 if cfg.splitType in [pd.SplitType.LOP, pd.SplitType.PARTIAL] else expectedNumPart
        eq(cfg.inSplit.getNumPartitions(), expectedIn)
        expectedOut = 1 if cfg.splitType in [pd.SplitType.LIP, pd.SplitType.PARTIAL] else expectedNumPart
        eq(cfg.outSplit.getNumPartitions(), expectedOut)

        for splitT in [cfg.inSplit, cfg.outSplit]:
            for splitAx in splitT.axes:
                maxVal = splitT.shape[splitAx.axis]
                for r in splitAx.ranges:
                    less(r[0], r[1], "Invalid range order")
                    geq(r[0], 0)
                    leq(r[0], maxVal)
                    geq(r[1], 0)
                    leq(r[1], maxVal)
                if i == len(path) - 1 and splitT == cfg.outSplit:
                    eq(splitAx.ranges[0][0], 0, "Last cfg must produce beginning")
                    eq(splitAx.ranges[-1][1], maxVal, "Last cfg must produce end")
                    for i, r in enumerate(splitAx.ranges):
                        if i == 0:
                            continue
                        eq(r[0], splitAx.ranges[i - 1][1], "Last cfg cannot have overlaps or gaps between partitions")


def opsStrToModAndSplitPath(ops):
    prevExpr = relayOp(ops[0], None)
    exprs = [prevExpr]
    opIndexToExpr = {0: prevExpr}
    for i, op in enumerate(ops[1:]):
        if isinstance(op, tuple):
            prevIndex = op[1]
            op = op[0]
            if isinstance(prevIndex, list):
                prevExpr = [opIndexToExpr[j] for j in prevIndex]
            else:
                prevExpr = opIndexToExpr[prevIndex]
        prevExpr = relayOp(op, prevExpr)
        exprs.append(prevExpr)
        opIndexToExpr[i + 1] = exprs[-1]

    mod = tvm.IRModule.from_expr(exprs[-1])
    mod = relay.transform.InferType()(mod)
    analyzer = graph_analyzer.GraphAnalyzer()
    analyzer.run(mod["main"])
    sn = analyzer.makeNet()

    maxOpSize = 0
    maxOp = None
    for op in sn.ops:
        if op.getSize() > maxOpSize:
            maxOpSize = op.getSize()
            maxOp = op
    assert maxOp != None
    targetExpr = analyzer.getExprFromBuf(maxOp.getOutputs()[0])

    discovery = pd.PathDiscovery(targetExpr, analyzer, sn)
    return mod, discovery.discoverBest(mod)
