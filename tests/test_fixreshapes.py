import testutil as tu
import helper_passes
from tvm import relay


class CountFns(relay.ExprVisitor):
    def __init__(self):
        super().__init__()

    def get(self, mod):
        self.count = 0
        self.visit(mod["main"])
        return self.count

    def visit_function(self, fn):
        self.count += 1
        return super().visit_function(fn)


def test(ops):
    print("-----", ops, "-----")
    mod = tu.opsStrToMod(ops)
    mod = relay.transform.FuseOps()(mod)
    countBefore = CountFns().get(mod)
    mod = helper_passes.FixReshapesPass()(mod)
    countAfter = CountFns().get(mod)
    tu.less(countAfter, countBefore)


test(["100", "dense1000", "reshape", "dense10"])
test(["100", "dense1000", "reshape", "dense10", "reshape", "dense10"])
test(["100", "dense1000", "reshape", "dense10", ("dense1000", 0), "reshape", "dense10", ("cat", [3, 6]), "dense10"])
