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
#import tvm
#from tvm import relay


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
