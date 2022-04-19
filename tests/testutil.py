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

# import tvm
# from tvm import relay


__anyFail = False
__tc = unittest.TestCase()
__tc.failureException
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def __report():
    if __anyFail:
        print("Test failure!")
        sys.exit(1)
    else:
        print("All tests OK!")


atexit.register(__report)


def __handle_uncaught(t, v, tb):
    fail("Uncaught exception!")
    sys.__excepthook__(t, v, tb)


sys.excepthook = __handle_uncaught

ass = __tc.assertTrue
eq = __tc.assertEqual
neq = __tc.assertNotEqual
inst = __tc.assertIsInstance
raises = __tc.assertRaises


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
    for sched in nx.all_topological_sorts(G):
        memTest = getSchedMemUsage(sched, G)
        allSched.append((sched, memTest))
        if memTest < minTested:
            minTested = memTest
            minSched = sched
    if minTested < memFound:
        fail(
            "Test failed: A better schedule was found: "
            + str(memFound)
            + " "
            + str(minTested)
            + " "
            + "".join(minSched)
        )
