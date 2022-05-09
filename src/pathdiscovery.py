from enum import Enum

from tvm import relay

import relay_util
import network
import memplanner
from optypes import OpArgType, OpType, getOpArgType, getOpType


MAX_PARTITIONS = 10
# Describes how an operation is split.
SplitType = Enum("SplitType", "NOTSPLIT PARTITIONED LOP LIP PARTIAL FTP")
# Describes the result of split inference. Adjusted means that the inferred from type was changed.
InferStatus = Enum("InferStatus", "VALID ADJUSTED INVALID")


def checkRanges(ranges):
    assert True not in [type(r[0]) != int or type(r[1]) != int for r in ranges]


# Returns the ranges of partitions, accoring to np.array_split. Represented as array of tuples.
def getSplitRanges(totalSize, numPartitions):
    assert totalSize >= numPartitions

    sectionSize, extraSizeSections = divmod(totalSize, numPartitions)
    ranges = []
    for i in range(numPartitions):
        if i < extraSizeSections:
            ranges.append((i * (sectionSize + 1), (i + 1) * (sectionSize + 1)))
        else:
            offset = extraSizeSections * (sectionSize + 1) + (i - extraSizeSections) * sectionSize
            ranges.append((offset, offset + sectionSize))
    return ranges


def applyFactorToRanges(ranges, factor, maxVal):
    out = []
    for r in ranges:
        start = max(0, factor * round(r[0] / factor))
        end = min(maxVal, factor * round(r[1] / factor))
        if start == end:
            # Partition was squeezed out completely.
            return None
        out.append((start, end))
    return out


# Contains all information about how splitting is done along an axis.
class SplitAxis:
    def __init__(self, axis, ranges):
        self.axis = axis
        self.ranges = ranges

    def clone(self):
        # Shallow copy is fine here, because tuples are not mutable.
        return SplitAxis(self.axis, self.ranges.copy())

    def getPartOffset(self, partNum):
        return self.ranges[partNum][0]

    # Returns the past-the-end index for the given partition.
    def getPartEnd(self, partNum):
        return self.ranges[partNum][1]

    def getNumPartElements(self, partNum):
        return self.getPartEnd(partNum) - self.getPartOffset(partNum)

    # Returns the number of elements in the largest partiton.
    def getMaxNumPartElements(self):
        return max([r[1] - r[0] for r in self.ranges])

    def getNumPartitions(self):
        return len(self.ranges)

    def __eq__(self, other):
        if isinstance(other, SplitAxis):
            return self.axis == other.axis and self.ranges == other.ranges
        return False

    def __repr__(self):
        return f"SplitAxis({self.axis}, {self.ranges})"


# Contains all information about how splitting is done for a tensor.
class SplitTensor:
    def __init__(self, shape, axes):
        assert len(shape) >= len(axes)
        self.shape = shape
        self.axes = axes

    def clone(self):
        return SplitTensor(self.shape, self.cloneAxes())

    def cloneAxes(self):
        return [ax.clone() for ax in self.axes]

    # Returns the shape of the split tensor for the given partition.
    def getSplitShape(self, partNum):
        # outShape = self.shape
        # for splitAx in self.axes:
        #     outShape[splitAx.axis] = splitAx.getNumPartElements(partNum)

        if len(self.axes) == 0:
            return self.shape
        elif len(self.axes) == 1:
            splitAx = self.axes[0]
            outShape = list(self.shape)
            outShape[splitAx.axis] = splitAx.getNumPartElements(partNum)
            return tuple(outShape)
        elif len(self.axes) == 2:
            ax1 = self.axes[0]
            ax2 = self.axes[1]
            outShape = list(self.shape)
            outShape[ax1.axis] = ax1.getNumPartElements(partNum // len(ax2.ranges))
            outShape[ax2.axis] = ax2.getNumPartElements(partNum % len(ax2.ranges))
            return tuple(outShape)

    def partNumToAxisIndex(self, partNum, axis):
        if len(self.axes) == 1:
            return partNum
        elif len(self.axes) == 2:
            if axis == self.axes[0].axis:
                return partNum // len(self.axes[1].ranges)
            elif axis == self.axes[1].axis:
                return partNum % len(self.axes[1].ranges)
            else:
                raise RuntimeError("invalid dim")
        else:
            raise RuntimeError("invalid number of axes")

    def partNumToRange(self, partNum, axis):
        if len(self.axes) == 2:
            index = self.partNumToAxisIndex(partNum, axis)
            if axis == self.axes[0].axis:
                return self.axes[0].ranges[index]
            elif axis == self.axes[1].axis:
                return self.axes[1].ranges[index]
            else:
                raise RuntimeError("invalid dim")
        elif len(self.axes) > 2:
            raise RuntimeError("cannot handle larger than 2d")
        else:
            return self.axes[0].ranges[partNum]

    def getNumPartitions(self):
        numPart = 1
        for splitAx in self.axes:
            numPart *= splitAx.getNumPartitions()
        return numPart

    # Returns the number of elements in the largest partition.
    def getMaxNumPartElements(self):
        numElems = 1
        for i, dim in enumerate(self.shape):
            found = False
            for splitAx in self.axes:
                if splitAx.axis == i:
                    numElems *= splitAx.getMaxNumPartElements()
                    found = True
            if not found:
                numElems *= dim
        return numElems

    def __eq__(self, other):
        if other == None:
            return False
        return self.shape == other.shape and self.axes == other.axes

    def __repr__(self):
        return f"SplitTensor({self.shape}, {self.axes})"


# Contains all information about how splitting is done for an expression.
class SplitConfig:
    def __init__(self, expr, splitType, inSplit=None, outSplit=None):
        assert inSplit != None or outSplit != None
        self.expr = expr
        self.splitType = splitType
        self.inSplit = inSplit
        self.outSplit = outSplit
        self.wSplit = None

    def clone(self):
        inSplit = self.inSplit.clone() if self.inSplit else None
        outSplit = self.outSplit.clone() if self.outSplit else None
        out = SplitConfig(self.expr, self.splitType, inSplit, outSplit)
        out.wSplit = self.wSplit.clone() if self.wSplit else None
        return out

    def inferUp(self):
        # print("inferUp", relay_util.exprToStr(self.expr))
        if self.splitType == SplitType.NOTSPLIT:
            inShape = relay_util.getShape(self.expr.args[0])
            self.inSplit = SplitTensor(inShape, [])
            return InferStatus.VALID
        elif self.splitType == SplitType.LIP:
            # Inferring up through LIP can only occur if we have already inferred it down, so skip.
            assert self.inSplit != None
            return InferStatus.VALID

        opArgTy = getOpArgType(self.expr)
        name = self.expr.op.name
        if opArgTy in [OpArgType.TRIVIAL, OpArgType.SIMPLE]:
            # handle special cases: pooling, padding
            if name == "nn.max_pool2d" or name == "nn.avg_pool2d":
                return self.inferPool(True)
            elif name == "nn.pad":
                padding = self.expr.attrs["pad_width"]
                padding = {i: (int(p[0]), int(p[1])) for i, p in enumerate(padding)}
                return self.inferComplex(True, padding=padding)
                inShape = relay_util.getShape(self.expr.args[0])
                newAxes = []
                for splitAx in self.outSplit.axes:
                    padBegin = int(padding[splitAx.axis][0])
                    padEnd = int(padding[splitAx.axis][1])
                    maxVal = inShape[splitAx.axis]
                    newRanges = []
                    for r in splitAx.ranges:
                        r = (r[0] - padBegin, r[1] - padBegin)
                        newRanges.append((r[0], r[1]))
                    newRanges[0] = (newRanges[0][0] + padBegin, newRanges[0][1])
                    newRanges[-1] = (newRanges[-1][0], newRanges[-1][1] - padEnd)
                    for r in newRanges:
                        if r[0] < 0 or r[1] > maxVal:
                            # While this could be bounded, we are at a huge overlap, so stop.
                            return InferStatus.INVALID
                    newAxes.append(SplitAxis(splitAx.axis, newRanges))
                self.inSplit = SplitTensor(inShape, newAxes)
            elif name == "mean":
                meanAxes = self.expr.attrs["axis"]
                assert len(meanAxes) == 1
                meanAxis = int(meanAxes[0])
                inShape = relay_util.getShape(self.expr.args[0])
                newAxes = []
                for splitAx in self.outSplit.axes:
                    newAxis = splitAx.axis
                    if splitAx.axis >= meanAxis:
                        newAxis += 1
                    newAxes.append(SplitAxis(newAxis, splitAx.ranges.copy()))
                self.inSplit = SplitTensor(inShape, newAxes)
            elif name == "take":
                assert isinstance(self.expr.args[0], relay.Constant)
                inShape = relay_util.getShape(self.expr.args[1])
                wShape = relay_util.getShape(self.expr.args[0])
                takeAxis = int(self.expr.attrs["axis"])
                newAxes = []
                wAxes = []
                for splitAx in self.outSplit.axes:
                    if splitAx.axis >= len(inShape):
                        wAxis = splitAx.axis - len(inShape)
                        if wAxis >= takeAxis:
                            wAxis += 1
                        wAxes.append(SplitAxis(wAxis, splitAx.ranges.copy()))
                    else:
                        newAxes.append(SplitAxis(splitAx.axis, splitAx.ranges.copy()))
                self.inSplit = SplitTensor(inShape, newAxes)
                self.wSplit = SplitTensor(wShape, wAxes)
            else:
                self.inSplit = self.outSplit.clone()
        elif opArgTy == OpArgType.DENSE:
            assert self.splitType == SplitType.LOP

            # Valid split amounts are restricted by batching.
            wShape = relay_util.getShape(self.expr.args[1])
            numBatches = wShape[0]
            if numBatches < self.getNumPartitions():
                return InferStatus.INVALID

            splitAx = self.outSplit.axes[0]
            if len(wShape) == 3 and self.inSplit == None:
                outRanges = applyFactorToRanges(splitAx.ranges, wShape[2], self.outSplit.shape[splitAx.axis])
                if outRanges == None:
                    return InferStatus.INVALID
            else:
                outRanges = splitAx.ranges

            if len(wShape) == 3:
                batchSize = wShape[2]
                wRanges = [(r[0] // batchSize, r[1] // batchSize) for r in outRanges]
            else:
                wRanges = outRanges
            self.wSplit = SplitTensor(wShape, [SplitAxis(0, wRanges)])
            newOutSplit = SplitTensor(self.outSplit.shape, [SplitAxis(splitAx.axis, outRanges)])

            inShape = relay_util.getShape(self.expr.args[0])
            self.inSplit = SplitTensor(inShape, [])

            if newOutSplit != self.outSplit:
                self.outSplit = newOutSplit
                return InferStatus.ADJUSTED
        elif opArgTy == OpArgType.DWCONV:
            # can only split by channels?!
            # how does it interact with FTP? -> should be fine to integrate in path
            inShape = relay_util.getShape(self.expr.args[0])
            if self.splitType == SplitType.PARTITIONED:
                self.inSplit = SplitTensor(inShape, self.outSplit.cloneAxes())
            elif self.splitType == SplitType.FTP:
                return self.inferFTP(True)
            elif self.splitType == SplitType.PARTIAL:
                self.inSplit = SplitTensor(inShape, [])
            else:
                raise RuntimeError("unexpected split type")
        elif opArgTy == OpArgType.CONV:
            if self.splitType == SplitType.LOP:
                inShape = relay_util.getShape(self.expr.args[0])
                self.inSplit = SplitTensor(inShape, [])
            else:
                return self.inferFTP(True)
        elif opArgTy == OpArgType.RESHAPE:
            inShape = relay_util.getShape(self.expr.args[0])
            outShape = relay_util.getShape(self.expr)
            if inShape == outShape:
                self.inSplit = SplitTensor(outShape, self.outSplit.cloneAxes())
            elif relay_util.isFlatten(self.expr):
                self.inSplit = SplitTensor(outShape, self.outSplit.cloneAxes())
            elif len(inShape) == len(outShape):
                # relay.reshape(relay.var(shape=))
                print(inShape, outShape)
                raise RuntimeError("same shape dim nums")
            # for splitAx in self.outSplit.axes:
            # for r in splitAx.ranges:
            # np.unravel_index(np.ravel_multi_index(IDX, outShape), inShape)
            # Set overlay shape!
            self.inSplit = SplitTensor(outShape, self.outSplit.cloneAxes())
        else:
            raise RuntimeError("unhandled op arg type")

        return InferStatus.VALID

    def inferDown(self):
        # print("inferDown", relay_util.exprToStr(self.expr))
        opArgTy = getOpArgType(self.expr)
        name = self.expr.op.name
        if opArgTy in [OpArgType.TRIVIAL, OpArgType.SIMPLE]:
            # handle special cases: pooling, padding
            if name == "nn.max_pool2d" or name == "nn.avg_pool2d":
                return self.inferPool(False)
            elif name == "nn.pad":
                padding = self.expr.attrs["pad_width"]
                padding = {i: (int(p[0]), int(p[1])) for i, p in enumerate(padding)}
                return self.inferComplex(False, padding=padding)
                newAxes = []
                for splitAx in self.inSplit.axes:
                    padBegin = int(padding[splitAx.axis][0])
                    padEnd = int(padding[splitAx.axis][1])
                    newRanges = []
                    for r in splitAx.ranges:
                        r = (r[0] + padBegin, r[1] + padBegin)
                        newRanges.append((r[0], r[1]))
                    newRanges[0] = (newRanges[0][0] - padBegin, newRanges[0][1])
                    newRanges[-1] = (newRanges[-1][0], newRanges[-1][1] + padEnd)
                    newAxes.append(SplitAxis(splitAx.axis, newRanges))
                outShape = relay_util.getShape(self.expr)
                self.outSplit = SplitTensor(outShape, newAxes)
            elif name == "mean":
                meanAxes = self.expr.attrs["axis"]
                assert len(meanAxes) == 1
                meanAxis = int(meanAxes[0])
                outShape = relay_util.getShape(self.expr)
                newAxes = []
                for splitAx in self.inSplit.axes:
                    newAxis = splitAx.axis
                    if splitAx.axis == meanAxis:
                        return InferStatus.INVALID
                    elif splitAx.axis > meanAxis:
                        newAxis -= 1
                    newAxes.append(SplitAxis(newAxis, splitAx.ranges.copy()))
                self.outSplit = SplitTensor(outShape, newAxes)
            elif name == "take":
                # assert isinstance(self.expr.args[0], relay.Constant)
                # inShape = relay_util.getShape(self.expr.args[1])
                # takeAxis = int(self.expr.attrs["axis"])
                # outShape = relay_util.getShape(self.expr)
                # newAxes = []
                # for splitAx in self.inSplit.axes:
                #     if splitAx.axis >= len(inShape):
                #         wAxis = splitAx.axis - len(inShape)
                #         if wAxis == takeAxis:
                #             return InferStatus.INVALID
                #     else:
                #         newAxes.append(SplitAxis(splitAx.axis, splitAx.ranges.copy()))
                # self.outSplit = SplitTensor(outShape, newAxes)
                pass
            else:
                self.outSplit = self.inSplit.clone()
        elif opArgTy == OpArgType.DENSE:
            assert self.splitType == SplitType.LIP
            outShape = relay_util.getShape(self.expr)
            self.outSplit = SplitTensor(outShape, [])
        elif opArgTy == OpArgType.DWCONV:
            # can only split by channels?!
            # how does it interact with FTP? -> should be fine to integrate in path
            inShape = relay_util.getShape(self.expr.args[0])
            if self.splitType == SplitType.PARTITIONED:
                self.outSplit = SplitTensor(inShape, self.inSplit.cloneAxes())
            elif self.splitType == SplitType.FTP:
                return self.inferFTP(False)
            elif self.splitType == SplitType.PARTIAL:
                self.outSplit = SplitTensor(inShape, [])
            else:
                raise RuntimeError("unexpected split type")
        elif opArgTy == OpArgType.CONV:
            if self.splitType == SplitType.LIP:
                outShape = relay_util.getShape(self.expr)
                self.outSplit = SplitTensor(outShape, [])
            else:
                return self.inferFTP(False)
        elif opArgTy == OpArgType.RESHAPE:
            inShape = relay_util.getShape(self.expr.args[0])
            outShape = relay_util.getShape(self.expr)
            if inShape == outShape:
                self.outSplit = SplitTensor(inShape, self.inSplit.cloneAxes())
            elif relay_util.isFlatten(self.expr):
                # TODO: this is too complex for now.
                return InferStatus.INVALID
                self.outSplit = SplitTensor(
                    inShape, self.inSplit.cloneAxes()
                )  ## keep the input shape! TODO: check if used correctly
                # TODO: when converting weights, these might be useful: ravel_multi_index, unravel_index
            elif len(inShape) == len(outShape):
                # relay.reshape(relay.var(shape=))
                print(inShape, outShape)
                raise RuntimeError("same shape dim nums")
        else:
            raise RuntimeError("unhandled op arg type")

        return InferStatus.VALID

    def inferFTP(self, isUpwards):
        widthAx = self.expr.attrs["data_layout"].index("W")
        heightAx = self.expr.attrs["data_layout"].index("H")
        wShape = relay_util.getShape(self.expr.args[1])
        kSize = {
            widthAx: wShape[self.expr.attrs["kernel_layout"].index("W")],
            heightAx: wShape[self.expr.attrs["kernel_layout"].index("H")],
        }
        strideWidthIndex = 1  # TODO: verify
        strideHeightIndex = 0
        strides = {
            widthAx: int(self.expr.attrs["strides"][strideWidthIndex]),
            heightAx: int(self.expr.attrs["strides"][strideHeightIndex]),
        }
        padding = relay_util.normalizePadding(self.expr.attrs["padding"])
        padding = {
            widthAx: (relay_util.getNormalizedPaddingPair(padding, True)),
            heightAx: (relay_util.getNormalizedPaddingPair(padding, False)),
        }
        return self.inferComplex(isUpwards, kSize, strides, padding)

    def inferPool(self, isUpwards):
        widthAx = self.expr.attrs["layout"].index("W")
        heightAx = self.expr.attrs["layout"].index("H")
        kernelWidthIndex = 1  # TODO verify
        kernelHeightIndex = 0
        poolSize = {
            widthAx: int(self.expr.attrs["pool_size"][kernelWidthIndex]),
            heightAx: int(self.expr.attrs["pool_size"][kernelHeightIndex]),
        }
        strides = {
            widthAx: int(self.expr.attrs["strides"][kernelWidthIndex]),
            heightAx: int(self.expr.attrs["strides"][kernelHeightIndex]),
        }
        padding = relay_util.normalizePadding(self.expr.attrs["padding"])
        padding = {
            widthAx: (relay_util.getNormalizedPaddingPair(padding, True)),
            heightAx: (relay_util.getNormalizedPaddingPair(padding, False)),
        }
        return self.inferComplex(isUpwards, poolSize, strides, padding)

    # def inferComplex(self, isUpwards, opSize, strides, padding, widthAx, heightAx):
    def inferComplex(self, isUpwards, opSize={}, strides={}, padding={}):
        opSize = {ax: ((sz - 1) // 2) + (sz // 2) for ax, sz in opSize.items()}

        if not isUpwards:
            if self.outSplit == None:
                # only infer once for conv  OR!!!: use pool logic for propagating conv! -> no, too uneven
                newAxes = []
                outShape = relay_util.getShape(self.expr)
                for splitAx in self.inSplit.axes:
                    if splitAx.getNumPartitions() > outShape[splitAx.axis]:
                        return InferStatus.INVALID
                    newRanges = getSplitRanges(outShape[splitAx.axis], splitAx.getNumPartitions())
                    newAxes.append(SplitAxis(splitAx.axis, newRanges))
                self.outSplit = SplitTensor(outShape, newAxes)

        inShape = relay_util.getShape(self.expr.args[0])
        newAxes = []
        for splitAx in self.outSplit.axes:
            maxVal = inShape[splitAx.axis]
            if splitAx.axis in opSize:
                baseOffset = opSize[splitAx.axis]
            else:
                baseOffset = 0
            if splitAx.axis in strides:
                strideFactor = strides[splitAx.axis]
            else:
                strideFactor = 1
            if splitAx.axis in padding:
                padBegin = padding[splitAx.axis][0]
                padEnd = padding[splitAx.axis][1]
            else:
                padBegin = 0
                padEnd = 0
            offset = baseOffset - padBegin - padEnd
            newRanges = []
            for r in splitAx.ranges:
                newR = (r[0] * strideFactor, (r[1] * strideFactor) - (strideFactor - 1))
                if r[0] != 0:
                    newR = (newR[0] - padBegin, newR[1])
                if r[1] != self.outSplit.shape[splitAx.axis]:
                    newR = (newR[0], newR[1] + padEnd)
                newRanges.append((newR[0], newR[1] + offset))
                if newRanges[-1][0] < 0 or newRanges[-1][1] > maxVal:
                    return InferStatus.INVALID
            newAxes.append(SplitAxis(splitAx.axis, newRanges))
        newInSplit = SplitTensor(inShape, newAxes)
        if self.inSplit != None and newInSplit != self.inSplit:
            status = InferStatus.ADJUSTED
        else:
            status = InferStatus.VALID
        self.inSplit = newInSplit
        return status

    def isImplicitSplit(self):
        return self.splitType == SplitType.LOP

    def isPartial(self):
        return self.splitType in [SplitType.LIP, SplitType.PARTIAL]

    # Returns information on how the given argument should be split.
    def splitWeights(self, argIndex):
        if self.wSplit != None:
            return self.wSplit

        arg = self.expr.args[argIndex]
        shape = relay_util.getShape(arg)
        opArgTy = getOpArgType(self.expr)

        if opArgTy == OpArgType.TRIVIAL:
            return SplitTensor(shape, [])
        elif opArgTy == OpArgType.SIMPLE:
            assert self.splitType not in [SplitType.LOP, SplitType.LIP]
            if isinstance(arg, relay.Constant):
                wSplitAxes = []
                for splitAx in self.outSplit.axes:
                    if len(shape) >= splitAx.axis and shape[splitAx.axis] != 1:
                        assert shape[splitAx.axis] == self.outSplit.shape[splitAx.axis]
                        wSplitAxes.append(splitAx)
                    # Otherwise, the axis is broadcast, so no need to split.
                return SplitTensor(shape, wSplitAxes)
        elif opArgTy == OpArgType.DWCONV:
            assert self.splitType not in [SplitType.LOP, SplitType.LIP]
            # Can only split by channel axis.
            for splitAx in self.outSplit.axes:
                if splitAx.axis == 3:
                    return SplitTensor(shape, [SplitAxis(self.expr.attrs["kernel_layout"].index("O"), splitAx.ranges)])
            return SplitTensor(shape, [])

        if self.splitType == SplitType.PARTITIONED:
            if opArgTy == OpArgType.RESHAPE:
                raise RuntimeError("NOTYET")
        elif self.splitType == SplitType.LOP:
            if opArgTy == OpArgType.DENSE:
                return self.wSplit
            elif opArgTy == OpArgType.CONV:
                assert len(self.outSplit.axes) == 1
                axis = SplitAxis(self.expr.attrs["kernel_layout"].index("O"), self.outSplit.axes[0].ranges)
                return SplitTensor(shape, [axis])
        elif self.splitType == SplitType.LIP:
            assert len(self.inSplit.axes) == 1
            if opArgTy == OpArgType.DENSE:
                axis = SplitAxis(1, self.inSplit.axes[0].ranges)
                return SplitTensor(shape, [axis])
            elif opArgTy == OpArgType.CONV:
                axis = SplitAxis(self.expr.attrs["kernel_layout"].index("I"), self.inSplit.axes[0].ranges)
                return SplitTensor(shape, [axis])
        elif self.splitType == SplitType.FTP:
            if opArgTy == OpArgType.CONV:
                # For FTP, the full kernel is required in every partition.
                return SplitTensor(shape, [])
        raise RuntimeError("not implemented " + str(self.splitType) + " " + str(opArgTy))

    def getInSize(self):
        return relay_util.getSize(relay_util.getCallInput(self.expr))

    def getOutSize(self):
        return relay_util.getSize(self.expr)

    def getOpSize(self):
        return self.getInSize() + self.getOutSize()

    def getInPartSize(self):
        return relay_util.getTypeSize(relay_util.getCallInput(self.expr)) * self.inSplit.getMaxNumPartElements()

    def getOutPartSize(self):
        return relay_util.getTypeSize(self.expr) * self.outSplit.getMaxNumPartElements()

    def getInPartSize_fix(self, partNum):
        size = 1
        for dim in self.inSplit.getSplitShape(partNum):
            size *= dim
        return size * relay_util.getTypeSize(relay_util.getCallInput(self.expr))

    def getOutPartSize_fix(self, partNum):
        size = 1
        for dim in self.outSplit.getSplitShape(partNum):
            size *= dim
        return size * relay_util.getTypeSize(self.expr)

    # This is not the exact size, because that may be different for every partition.
    def getPartSize(self):
        return self.getInPartSize() + self.getOutPartSize()
        # if self.splitType == SplitType.LOP:
        #     return self.getInSize() + self.getOutSize() / self.getNumPartitions()
        # elif self.splitType == SplitType.PARTITIONED:
        #     return (self.getInSize() + self.getOutSize()) / self.getNumPartitions()
        # elif self.splitType == SplitType.LIP:
        #     return self.getInSize() / self.getNumPartitions() + self.getOutSize()
        # elif self.splitType in [SplitType.PARTIAL, SplitType.NOTSPLIT]:
        #     return self.getInSize() + self.getOutSize()
        # elif self.splitType == SplitType.FTP:
        #     return (self.getInSize() + self.getOutSize()) / self.getNumPartitions() # TODO overlap!
        # else:
        #     raise RuntimeError("unexpected split type: " + str(self.splitType))

    def getNumPartitions(self):
        if self.splitType == SplitType.LIP:
            return self.inSplit.getNumPartitions()
        else:
            return self.outSplit.getNumPartitions()

    def __repr__(self) -> str:
        if self.inSplit == None or self.outSplit == None:
            numPart = "ERR"
        else:
            numPart = str(self.getNumPartitions())
        return (
            "SplitConfig("
            + numPart
            + ", "
            + relay_util.exprToStr(self.expr)
            + ", "
            + str(self.splitType)
            + ", "
            + str(self.outSplit)
            + ")"
        )


# Represents a chain of expressions with information how they are split.
class SplitPath:
    def __init__(self, baseCfg):
        self.baseCfg = baseCfg
        self.preCfgs = []
        self.postCfgs = []

    def clone(self):
        out = SplitPath(self.baseCfg.clone())
        out.preCfgs = [cfg.clone() for cfg in self.preCfgs]
        out.postCfgs = [cfg.clone() for cfg in self.postCfgs]
        return out

    # For the purpose of path de-duplication after pruning. Does not do a full comparison.
    def isEquivalent(self, other):
        if self.getNumPartitions() != other.getNumPartitions():
            return False
        if len(self) != len(other):
            return False
        for i, cfg in enumerate(self):
            if cfg.splitType != other[i].splitType:
                return False
            if len(cfg.inSplit.axes) != len(other[i].inSplit.axes):
                return False
            if len(cfg.outSplit.axes) != len(other[i].outSplit.axes):
                return False
        return True

    def addCfg(self, cfg, prepend):
        if prepend:
            self.preCfgs.insert(0, cfg)
            status = cfg.inferUp()
            if status == InferStatus.INVALID:
                return False
            wasAdjusted = False
            # print("+", str(self))
            # print("+++ first pass +++")
            for i in range(1, len(self)):
                self[i].inSplit = self[i - 1].outSplit
                status = self[i].inferDown()
                # print("+", str(self))
                if status == InferStatus.INVALID:
                    return False
                elif status == InferStatus.ADJUSTED:
                    wasAdjusted = True
            # print("+++ second pass +++")
            if wasAdjusted:
                for i in range(len(self) - 2, -1, -1):
                    self[i].outSplit = self[i + 1].inSplit
                    status = self[i].inferUp()
                    # print("+", str(self))
                    if status == InferStatus.INVALID:
                        return False
                    elif status == InferStatus.ADJUSTED:
                        print("while inferring up:", relay_util.exprToStr(self[i].expr), self)
                        raise RuntimeError("did not expect further adjustments")
            # print("+++ done +++")
        else:
            self.postCfgs.append(cfg)
            status = cfg.inferDown()
            if status == InferStatus.INVALID:
                return False
            wasAdjusted = False
            # print("~", str(self))
            # print("~~~ first pass ~~~")
            for i in range(len(self) - 2, -1, -1):
                self[i].outSplit = self[i + 1].inSplit
                status = self[i].inferUp()
                # print("~", str(self))
                if status == InferStatus.INVALID:
                    return False
                elif status == InferStatus.ADJUSTED:
                    wasAdjusted = True
            # print("~~~ second pass ~~~")
            if wasAdjusted:
                for i in range(1, len(self)):
                    self[i].inSplit = self[i - 1].outSplit
                    status = self[i].inferDown()
                    # print("~", str(self))
                    if status == InferStatus.INVALID:
                        return False
                    elif status == InferStatus.ADJUSTED:
                        print("while inferring down:", relay_util.exprToStr(self[i].expr), self)
                        raise RuntimeError("did not expect further adjustments")
            # print("~~~ done ~~~")

        return True

    def getNumPartitions(self):
        return self.baseCfg.getNumPartitions()

    # Prunes the path such that the beginning and ending node have the lowest size compared to other preceiding and following nodes of the base node.
    def limitToMinSize(self):
        if len(self.postCfgs) == 0:
            return False
        if len(self.preCfgs) != 0:
            preIdx = self.preCfgs.index(min(reversed(self.preCfgs), key=lambda x: x.getInSize()))
            self.preCfgs = self.preCfgs[preIdx:]

        # Cannot allow to cut off overlapping ops.
        earliestCut = 0
        for i, cfg in enumerate(reversed(self.postCfgs)):
            if relay_util.hasOverlappingInput(cfg.expr):
                earliestCut = len(self.postCfgs) - i
                break
        checkMinCfgs = self.postCfgs[earliestCut:]

        if len(checkMinCfgs) != 0:
            postIdx = self.postCfgs.index(min(checkMinCfgs, key=lambda x: x.getOutSize()))
            self.postCfgs = self.postCfgs[: postIdx + 1]

        return True

    def __len__(self):
        return len(self.preCfgs) + 1 + len(self.postCfgs)

    def getitemImpl(self, idx):
        idx %= len(self)
        if idx == len(self.preCfgs):
            return self.baseCfg
        elif idx < len(self.preCfgs):
            return self.preCfgs[idx]
        elif (idx - len(self.preCfgs) - 1) < len(self.postCfgs):
            return self.postCfgs[idx - len(self.preCfgs) - 1]
        else:
            raise IndexError("out of bounds")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.getitemImpl(i) for i in range(*idx.indices(len(self)))]
        return self.getitemImpl(idx)

    def __iter__(self):
        for cfg in self.preCfgs:
            yield cfg
        yield self.baseCfg
        for cfg in self.postCfgs:
            yield cfg

    def shortDesc(self):
        if len(self[0].outSplit.axes) == 2:
            numPart = self[0].outSplit.axes[0].getNumPartitions()
            numPart = str(numPart) + "x" + str(numPart)
        else:
            numPart = str(self.getNumPartitions())
        return "_".join([cfg.splitType.name for cfg in self]) + "_" + numPart

    def __repr__(self):
        out = "SplitPath(" + str(self.getNumPartitions()) + ",\n"
        for cfg in self.preCfgs:
            out += "  " + str(cfg) + "\n"
        out += " >" + str(self.baseCfg) + "\n"
        for cfg in self.postCfgs:
            out += "  " + str(cfg) + "\n"
        out += ")"
        return out


# Creates SplitConfigs for every reasonable option for splitting the expression.
def createAllSplitConfigs(expr):
    cfgs = []

    opTy = getOpType(expr)
    shape = relay_util.getShape(expr)

    # Do not split.
    if opTy != OpType.NONE:
        cfgs.append(SplitConfig(expr, SplitType.NOTSPLIT, outSplit=SplitTensor(shape, [])))

    # Split by every axis.
    if opTy in [OpType.ELEMWISE_LINEAR, OpType.ELEMWISE_NONLINEAR, OpType.DENSE, OpType.CONV, OpType.POOL]:
        for axis, dim in enumerate(shape):
            if dim < 2:
                # In TVM, the first axis is always used for batching, which is one for inference.
                continue

            if len(shape) == 4 and axis != 3 and opTy != OpType.DENSE:
                splitType = SplitType.FTP
                if axis == 2 and shape[1] == shape[2]:
                    # Skip equivalent configs.
                    continue
            elif opTy in [OpType.DENSE, OpType.CONV]:
                splitType = SplitType.LOP
            else:
                splitType = SplitType.PARTITIONED

            if expr.op.name == "take":
                assert isinstance(expr.args[0], relay.Constant)
                inShape = relay_util.getShape(expr.args[1])
                if axis >= len(inShape):
                    splitType = SplitType.LOP
                else:
                    splitType = SplitType.PARTITIONED

            maxPartitions = min(dim, MAX_PARTITIONS)
            for numPartitions in range(2, maxPartitions + 1):
                ranges = getSplitRanges(dim, numPartitions)
                cfgs.append(SplitConfig(expr, splitType, outSplit=SplitTensor(shape, [SplitAxis(axis, ranges)])))

    # Split by two axes, FTP style.
    if opTy in [OpType.ELEMWISE_LINEAR, OpType.ELEMWISE_NONLINEAR, OpType.CONV, OpType.POOL]:
        if len(shape) == 4 and shape[1] >= 3 and shape[2] >= 3:
            ranges1 = getSplitRanges(shape[1], 2)
            ranges2 = getSplitRanges(shape[2], 2)
            cfgs.append(
                SplitConfig(
                    expr, SplitType.FTP, outSplit=SplitTensor(shape, [SplitAxis(1, ranges1), SplitAxis(2, ranges2)])
                )
            )
            ranges1 = getSplitRanges(shape[1], 3)
            ranges2 = getSplitRanges(shape[2], 3)
            cfgs.append(
                SplitConfig(
                    expr, SplitType.FTP, outSplit=SplitTensor(shape, [SplitAxis(1, ranges1), SplitAxis(2, ranges2)])
                )
            )

    return cfgs


# Creates SplitConfigs for the given expr when it is added (front or back, based on isUpwards) to the given path.
def createChainedSplitConfigs(expr, path, isUpwards):
    outCfgs = []

    if expr == None:
        return [None]

    if isUpwards:
        fromCfg = path[0]
    else:
        fromCfg = path[-1]

    if fromCfg.splitType == SplitType.NOTSPLIT:
        return [None]

    if isUpwards and fromCfg.splitType == SplitType.LOP:
        # There cannot be any expr before LOP.
        return [None]

    if fromCfg.splitType == SplitType.FTP and relay_util.hasOverlappingInput(expr):
        # Emit one path that stops before the operation with overlaps.
        outCfgs.append(None)

    opTy = getOpType(expr)

    if not isUpwards and fromCfg.isPartial():
        if opTy != OpType.ELEMWISE_LINEAR:
            # Cannot proceed from non-linear while already switched to partial values.
            return [None]
        splitType = SplitType.PARTIAL
    elif opTy == OpType.NONE:
        # Splitting has to stop here because the next expr does not support splitting.
        return [None]
    elif opTy in [OpType.ELEMWISE_LINEAR, OpType.ELEMWISE_NONLINEAR, OpType.POOL]:
        splitType = fromCfg.splitType
        assert splitType != SplitType.LIP
        if splitType == SplitType.LOP:
            splitType = SplitType.PARTITIONED
    elif opTy in [OpType.DENSE, OpType.CONV]:
        if fromCfg.splitType == SplitType.FTP:
            splitType = SplitType.FTP
        elif isUpwards:
            if opTy == OpType.DENSE:
                wShape = relay_util.getShape(expr.args[1])
                if wShape[0] < fromCfg.getNumPartitions():
                    return []
            splitType = SplitType.LOP
        else:
            splitType = SplitType.LIP
            # Emit one path that stops here.
            outCfgs.append(None)
    else:
        raise RuntimeError("unhandled op type")

    if isUpwards:
        outCfgs.append(SplitConfig(expr, splitType, outSplit=fromCfg.inSplit))
    else:
        outCfgs.append(SplitConfig(expr, splitType, inSplit=fromCfg.outSplit))

    return outCfgs


class SplitOpInfo:
    def __init__(self, expr, splitType, isImplicitSplit=False, isPartial=False):
        self.expr = expr
        self.splitType = splitType
        self.isImplicitSplit = isImplicitSplit
        self.isPartial = isPartial

    def getOutSize(self):
        return relay_util.getSize(self.expr)

    def getInSize(self):
        return relay_util.getSize(relay_util.getCallInput(self.expr))

    def __repr__(self):
        return (
            "SplitOpInfo("
            + relay_util.exprToStr(self.expr)
            + ", "
            + str(self.splitType)
            + (", implicit" if self.isImplicitSplit else "")
            + (", partial" if self.isPartial else "")
            + ")"
        )


class PathDiscovery:
    def __init__(self, startExpr, analyzer, n, noFTP=False, onlyFTP=False):
        self.startExpr = startExpr
        self.analyzer = analyzer
        self.n = n
        self.splitInfos = []
        self.splitPaths = []
        self.noFTP = noFTP
        self.onlyFTP = onlyFTP
        assert not self.noFTP or not self.onlyFTP

    def discoverBest(self, mod):
        self.discoverAll()

        bestSize = self.analyzer.exprToOp[self.startExpr].getSize()
        bestPath = None
        import moiopt
        import graph_analyzer
        for path in self.splitPaths:
            print("PATH:", path)
            testMod = moiopt.SplitPathPass(path)(mod)
            testMod = relay.transform.InferType()(testMod)
            testMod = relay.transform.FuseOps()(testMod)
            testMod = moiopt.FixTupleDepdendencyPass()(testMod)

            analyzer = graph_analyzer.GraphAnalyzer()
            analyzer.run(testMod["main"])
            testNet = analyzer.makeNet()
            sched = testNet.createBestSchedule()
            planner = memplanner.MemoryPlanner(sched)

            def cbLayout(memLayout):
                if (
                    path[0].splitType == SplitType.FTP
                    and path[0].getNumPartitions() == 9
                    and len(path) == 13
                    and len(path[0].outSplit.axes) == 2
                ):
                    import plot

                    plot.drawLayoutChart("ftp3x3", memLayout, planner)
                return {testNet.bufs[0]: memLayout.getSize()}

            sz = memplanner.memLayoutWithTimeout(testNet, planner, cbLayout)
            sz = sz.popitem()[1]
            print(path.shortDesc(), "-----", sz)
            if sz < bestSize:
                bestSize = sz
                bestPath = path

        if bestPath != None and bestPath.getNumPartitions() < 2:
            return None

        return bestPath

    def discoverAll(self):
        baseCfgs = createAllSplitConfigs(self.startExpr)
        self.workingPaths = [SplitPath(cfg) for cfg in baseCfgs if cfg.inferUp() != InferStatus.INVALID]

        for path in self.workingPaths.copy():
            if path[0].splitType == SplitType.FTP:
                if self.noFTP:
                    self.workingPaths.remove(path)
            else:
                if self.onlyFTP:
                    self.workingPaths.remove(path)

        prevExpr = self.getPrevExpr(self.startExpr)
        while prevExpr and len(self.workingPaths) > 0:
            self.updateSplitPaths(prevExpr, True)
            prevExpr = self.getPrevExpr(prevExpr)

        self.workingPaths.extend(self.splitPaths)
        self.splitPaths = []

        nextExpr = self.getNextExpr(self.startExpr)
        while nextExpr and len(self.workingPaths) > 0:
            self.updateSplitPaths(nextExpr, False)
            nextExpr = self.getNextExpr(nextExpr)

        self.splitPaths.extend(self.workingPaths)

        # Prune paths to optimal split and merge positions.
        for path in self.splitPaths.copy():
            if not path.limitToMinSize():
                self.splitPaths.remove(path)

        # Limit the number of partitions if path starts at preexisting split.
        for path in self.splitPaths.copy():
            splittingOp = self.analyzer.exprToOp[path[0].expr]
            preds = list(self.n.g.predecessors(splittingOp))
            if len(preds) == 0:
                existingPartitions = len(self.n.getInOps())
            else:
                assert len(preds) == 1
                existingPartitions = self.n.g.out_degree(preds[0])
            if path.getNumPartitions() > (MAX_PARTITIONS + 1 - existingPartitions):
                self.splitPaths.remove(path)

        # De-duplicate paths. Duplicate paths occur from pruning.
        iterCopy = self.splitPaths.copy()
        for i, path in enumerate(iterCopy):
            for j, p in enumerate(iterCopy):
                if j >= i:
                    break
                if path.isEquivalent(p) and p in self.splitPaths:
                    self.splitPaths.remove(p)

        return self.splitPaths

    def updateSplitPaths(self, expr, isUpwards):
        for path in self.workingPaths.copy():
            newCfgs = createChainedSplitConfigs(expr, path, isUpwards)

            self.workingPaths.remove(path)
            for cfg in newCfgs:
                if cfg == None:
                    # Path stops here. Move it to final list.
                    self.splitPaths.append(path)
                else:
                    newPath = path.clone()
                    if newPath.addCfg(cfg, isUpwards):
                        self.workingPaths.append(newPath)

    def getPrevExpr(self, expr):
        op = self.analyzer.exprToOp[expr]
        preds = list(self.n.g.predecessors(op))
        if len(preds) != 1 or len(list(self.n.g.successors(preds[0]))) != 1:
            return None
        buf = preds[0].getOutputs()[0]
        return self.analyzer.bufToExpr[buf]

    def getNextExpr(self, expr):
        op = self.analyzer.exprToOp[expr]
        succs = list(self.n.g.successors(op))
        if len(succs) != 1 or len(list(self.n.g.predecessors(succs[0]))) != 1:
            return None
        buf = succs[0].getOutputs()[0]
        return self.analyzer.bufToExpr[buf]
