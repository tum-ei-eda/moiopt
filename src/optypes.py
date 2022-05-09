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
        "add",
        "subtract",
        "multiply",
        "divide",
        "fixed_point_multiply",
        "reshape",
        "right_shift",  # ???
        "nn.pad",
        "take",
    ]:
        return OpType.ELEMWISE_LINEAR
    elif name in ["round", "clip", "nn.relu", "tanh"]:
        return OpType.ELEMWISE_NONLINEAR
    elif name in [
        "nn.softmax",
        "strided_slice",
        "concatenate",
    ]:
        # TODO: concat is splitable and possible target of identity transformation
        return OpType.NONE
    elif name == "nn.contrib_dense_pack":
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
        "nn.relu",
        "nn.max_pool2d",
        "nn.avg_pool2d",
        "nn.pad",
        "mean",
    ]:
        return OpArgType.TRIVIAL
    elif name in ["add", "subtract", "multiply", "right_shift", "take"]:
        return OpArgType.SIMPLE
    elif name == "reshape":
        return OpArgType.RESHAPE
    elif name == "nn.conv2d":
        if relay_util.isDepthwiseConv(expr):
            return OpArgType.DWCONV
        else:
            return OpArgType.CONV
    elif name == "nn.contrib_dense_pack":
        return OpArgType.DENSE
    else:
        raise RuntimeError("unhandled op name: " + name)
