from collections import defaultdict
import networkx as nx
import layoutopt
import exec_timeout


def doesOverlap(start1, size1, start2, size2):
    return (start1 < (start2 + size2)) and (start2 < (start1 + size1))


class CriticalChain:
    def __init__(self, sizeLeft):
        self.sizeLeft = sizeLeft
        self.bufList = []

    def clone(self):
        cc = CriticalChain(self.sizeLeft)
        cc.bufList = self.bufList.copy()
        return cc


class MemoryLayout:
    def __init__(self):
        self.numBuckets = 0
        self.buckets = []
        self.bufToOffset = {}
        self.bufToBucket = {}
        self.bucketSizes = {}

    def newBucket(self):
        index = len(self.buckets)
        self.bucketSizes[index] = 0
        self.buckets.append([])
        return index

    def fitsIntoBucket(self, buf, offset, index):
        for otherBuf in self.buckets[index]:
            if doesOverlap(offset, buf.size, self.bufToOffset[otherBuf], otherBuf.size):
                return False

        return True

    def addBufToBucket(self, buf, index, offset=None):
        if offset != None:
            if not self.fitsIntoBucket(buf, offset, index):
                raise RuntimeError("WRONG BUFFER ASSIGNMENT")
        else:
            offset = self.bucketSizes[index]

        self.buckets[index].append(buf)
        self.bufToOffset[buf] = offset
        self.bufToBucket[buf] = index
        self.bucketSizes[index] = max(self.bucketSizes[index], buf.size + offset)
        # print("adding " + buf.name + " to bucket " + str(index) + " with offset " + str(offset))

    def addBufToOffset(self, buf, offset):
        # Find free bucket.
        freeBucket = None
        for i in range(0, len(self.buckets)):
            if self.fitsIntoBucket(buf, offset, i):
                freeBucket = i
                break

        if freeBucket == None:
            freeBucket = self.newBucket()

        self.addBufToBucket(buf, freeBucket, offset)

    def getOffset(self, buf):
        return self.bufToOffset[buf]

    def getBucket(self, buf):
        return self.bufToBucket[buf]

    def isPlaced(self, buf):
        return buf in self.bufToOffset

    def getSize(self):
        return max(self.bucketSizes.values(), default=0)

    # Returns all bufs that could be responsible for the peak memory usage, sorted by size.
    def getBufsByCriticality(self):
        criticalChains = [CriticalChain(self.getSize())]
        anythingFound = True
        while anythingFound:
            anythingFound = False
            nextChains = []
            for cc in criticalChains:
                if cc.sizeLeft == 0:
                    nextChains.append(cc)
                    continue
                for buf, offset in self.bufToOffset.items():
                    if offset + buf.size == cc.sizeLeft:
                        ncc = cc.clone()
                        ncc.bufList.append(buf)
                        ncc.sizeLeft -= buf.size
                        nextChains.append(ncc)
                        anythingFound = True
            criticalChains = nextChains
        criticalBufs = set()
        for cc in criticalChains:
            criticalBufs.update(cc.bufList)
        return sorted(list(criticalBufs), key=lambda b: b.size, reverse=True)

    def __repr__(self):
        out = "MemLayout():\n"
        for i, b in enumerate(self.buckets):
            out += "  - Bucket " + str(i) + "\n"
            for buf in b:
                out += "    - Buf: " + str(buf) + "\n"
        out += "  ----- sz: " + str(self.getSize())
        return out


class Lifetime:
    def __init__(self):
        self.firstUse = 0xFFFFFFFF
        self.lastUse = -1

    def addUse(self, t):
        self.firstUse = min(t, self.firstUse)
        self.lastUse = max(t, self.lastUse)


class Lifetimes:
    def __init__(self):
        self.ranges = defaultdict(Lifetime)
        self.ts = defaultdict(set)

    def addUse(self, t, key):
        lt = self.ranges[key]
        lt.addUse(t)
        for i in range(lt.firstUse, lt.lastUse + 1):
            self.ts[i].add(key)

    def getFirstUse(self, key):
        return self.ranges[key].firstUse

    def getLastUse(self, key):
        return self.ranges[key].lastUse

    def getDuration(self, key):
        return self.getLastUse(key) - self.getFirstUse(key) + 1

    def getKeysAt(self, t):
        return self.ts[t]


class MemoryPlanner:
    def __init__(self, sched):
        self.sched = sched

        self.conflictGraph = nx.Graph()
        self.lifetimes = Lifetimes()

        for i, op in enumerate(self.sched.sched):
            for inBuf in op.getInputs():
                if not inBuf.isStatic():
                    self.lifetimes.addUse(i, inBuf)
            for outBuf in op.getOutputs():
                self.lifetimes.addUse(i, outBuf)

        for t in range(0, len(self.sched.sched)):
            bufsAtT = list(self.lifetimes.getKeysAt(t))
            for i in range(0, len(bufsAtT)):
                for j in range(i + 1, len(bufsAtT)):
                    self.conflictGraph.add_edge(bufsAtT[i], bufsAtT[j])

    def createOptimalLayout(self):
        bufs = {}
        bufSizes = []
        for i, buf in enumerate(self.conflictGraph.nodes()):
            bufs[buf] = i
            bufSizes.append(buf.size)

        conflicts = []
        for buf, i in bufs.items():
            otherBufs = list(self.conflictGraph.neighbors(buf))
            for otherBuf in otherBufs:
                j = bufs[otherBuf]
                if i < j:
                    conflicts.append((i, j))

        offsets = layoutopt.solve(bufSizes, conflicts)

        memLayout = MemoryLayout()
        for buf, i in bufs.items():
            memLayout.addBufToOffset(buf, offsets[i])
        return memLayout

    def createTFLMLayout(self):
        memLayout = MemoryLayout()

        # Sort buffers by size descending.
        sortedBufs = sorted(self.conflictGraph.nodes(), key=lambda x: x.size, reverse=True)

        # Place first buffer.
        memLayout.newBucket()
        memLayout.addBufToBucket(sortedBufs[0], 0)

        # Loop through in descending order.
        for i in range(1, len(sortedBufs)):
            buf = sortedBufs[i]

            # Look for other buffers that need to be in memory at the same time.
            otherBufs = list(self.conflictGraph.neighbors(buf))

            # Use the first available gap.
            otherPlacedBufs = [b for b in otherBufs if memLayout.isPlaced(b)]
            otherPlacedBufs.sort(key=lambda x: memLayout.getOffset(x))
            currentOffset = 0
            while True:
                gapFound = True
                for otherBuf in otherPlacedBufs:
                    otherOffset = memLayout.getOffset(otherBuf)
                    if doesOverlap(currentOffset, buf.size, otherOffset, otherBuf.size):
                        currentOffset = max(currentOffset, otherOffset + otherBuf.size)
                        gapFound = False
                        break
                if gapFound:
                    break

            memLayout.addBufToOffset(buf, currentOffset)

        return memLayout


def memLayoutWithTimeout(n, planner, func, timeout=0.5, noILP=False):
    def inExternalProcess():
        bufsToIds = {buf: i for i, buf in enumerate(n.bufs)}
        memLayout = planner.createOptimalLayout()
        return {bufsToIds[buf]: data for buf, data in func(memLayout).items()}

    if not noILP:
        idsToBufs = {i: buf for i, buf in enumerate(n.bufs)}
        try:
            idsToData = exec_timeout.exec_timeout(timeout, inExternalProcess)
            return {idsToBufs[id]: data for id, data in idsToData.items()}
        except TimeoutError:
            pass

    memLayout = planner.createTFLMLayout()
    return {buf: data for buf, data in func(memLayout).items()}
