from enum import Enum

import tvm
from tvm import relay

import relay_util


OpType = Enum("OpType", "NONE ELEMWISE_LINEAR ELEMWISE_NONLINEAR SPLIT CONCAT MERGE DENSE CONV POOL")
OpArgType = Enum("OpArgType", "TRIVIAL SIMPLE RESHAPE DWCONV CONV DENSE")


def getOpType(expr):
    if not isinstance(expr, relay.Call):
        return OpType.NONE
    # We do not expect functions at this point.
    assert isinstance(expr.op, tvm.ir.Op)
    name = expr.op.name

    # TODO disallow any flatten ops for now for simplicity.
    if name == "reshape" and relay_util.isFlatten(expr):
        return OpType.NONE

    if name in [
        # "add",  # could be supported by other means.
        # "subtract",
        "multiply",
        "divide",
        "fixed_point_multiply",
        "reshape",
        "right_shift",  # ???
        "nn.pad",
        "take",
        "transpose",
        "expand_dims",
    ]:
        return OpType.ELEMWISE_LINEAR
    elif name in ["round", "clip", "nn.relu", "nn.prelu", "tanh", "sigmoid", "add", "subtract"]:
        return OpType.ELEMWISE_NONLINEAR
    elif name in [
        "nn.softmax",
        "strided_slice",
        "concatenate",
        "split",
        "nn.conv2d_transpose",  # TODO: trivial inversion of conv, but CONV optype might need rework
        "image.resize2d",  # TODO: could be treated as optype POOL, but high arg complexity
        "nn.depth_to_space",  # TODO: could be treated as optype POOL?
    ]:
        # TODO: concat is splitable and possible target of identity transformation
        return OpType.NONE
    elif name in ["nn.contrib_dense_pack", "nn.dense"]:
        return OpType.DENSE
    elif name == "nn.conv2d":
        if relay_util.isDepthwiseConv(expr):
            return OpType.ELEMWISE_LINEAR
        else:
            return OpType.CONV
    elif name in ["nn.max_pool2d", "nn.avg_pool2d", "mean"]:
        return OpType.POOL
    elif name == "cast":
        inTy = relay_util.RelayType(expr.args[0])
        outTy = relay_util.RelayType(expr)
        if inTy.isFloat() and outTy.isInt():
            # Precision loss.
            return OpType.ELEMWISE_NONLINEAR
        if inTy.getTypeSizeBits() > outTy.getTypeSizeBits():
            # Precision loss.
            return OpType.ELEMWISE_NONLINEAR
        return OpType.ELEMWISE_LINEAR
    else:
        raise RuntimeError("unhandled op name: " + name)


def getOpArgType(expr):
    assert isinstance(expr, relay.Call)
    assert isinstance(expr.op, tvm.ir.Op)
    name = expr.op.name

    if name in [
        "clip",
        "cast",
        "fixed_point_multiply",
        "tanh",
        "sigmoid",
        "nn.relu",
        "nn.prelu",
        "nn.max_pool2d",
        "nn.avg_pool2d",
        "nn.pad",
        "mean",
        "transpose",
        "round",
        "expand_dims",
    ]:
        return OpArgType.TRIVIAL
    elif name in ["add", "subtract", "multiply", "divide", "right_shift", "take"]:
        return OpArgType.SIMPLE
    elif name == "reshape":
        return OpArgType.RESHAPE
    elif name == "nn.conv2d":
        if relay_util.isDepthwiseConv(expr):
            return OpArgType.DWCONV
        else:
            return OpArgType.CONV
    elif name in ["nn.contrib_dense_pack", "nn.dense"]:
        return OpArgType.DENSE
    else:
        raise RuntimeError("unhandled op name: " + name)
