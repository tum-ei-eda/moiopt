import random
import testutil as tu
import schedule


class Example:
    def __init__(self, sched):
        self.sched = sched


def testSP(*args, expect=True):
    print("=== testSP", args, "===")
    G = tu.makeGraph(*args)
    isSP = schedule.is_sp_graph(G)
    tu.eq(isSP, expect)


def test(*args, expect=None, weights={}):
    print("-----", *args, "-----")

    G = tu.makeGraph(*args, weights=weights)

    opt = schedule.SchedOpt()
    sched = opt.solve(G)
    sched = "".join(sched)

    if expect == None:
        tu.verifySched(sched, G)
    elif isinstance(expect, Example):
        tu.eq(tu.getSchedMemUsage(sched, G), tu.getSchedMemUsage(expect.sched, G))
    else:
        tu.eq(sched, expect)


testSP("")
testSP("a")
testSP("ab")
testSP("abcde")
testSP("abc", "aBc")
testSP("abc", "aBc", "aXc")
testSP("ab", "Xb", expect=False)
testSP("abcd", "aBCd", "bC", expect=False)
testSP("abcde", "aBCDe", "BXD", "bYd")
testSP("abcde", "aBcDe")
testSP("abcde", "aBcDe", "aXYc")
testSP("abcdefg", "aBCdeFg")
testSP("abcd", "aBc", "aXYd")
testSP("abcd", "aBCd", "Bc", expect=False)
testSP("abcd", "aBCd", "Bc", "ad", expect=False)
testSP("abc", "ac")


# Simple diamond.
test("abc", "aBc")
# Diamond in chain.
test("abcde", "abCde", "abcCde")
# 2 parallel in chain.
test("abcdef", "abCDef")
# 3 parallel.
test("abc", "aBc", "aXc")
# Consecutive.
test("abcde", "aBcDe")
# Nested.
test("abcd", "aBc", "aXd")
test("abcd", "aBCd", "aBXd")
test("abcd", "aBCd", "aXCd")
test("abcdme", "bCd", "aXMe")
test("abcdme", "bCd", "aXMe", weights={"X": 1000})
# Multi input.
test("ab", "Ab")
test("abc", "ABc")
test("abcd", "aBCd", "XYd")
test("abcd", "aBcd", "Xcd")
test("abcd", "aBcd", "Xd")
test("ab", "Ab", "XZb", "xZ")
# Stress test.
test("")
test("a")
test("a", "b")
test("ab")
test("abc")
test("ab", "aB")
test("abc", "aBc", "Xc")
test("abc", "aBc", "aX")
test("abc", "aBc", "bX")
test("abcdefgh", "cDEFg", "gh")
# Non-SP graph.
test("abcd", "aBCd", "bC")


test("abcdefghijk", "abcDEFg", "mnopqjk", expect=Example("abcDEFdefghimnopqjk"))
# test(
#     "abcdefghi",
#     "aBCDEFGh",
#     "CZYXVh",
#     "YxV",
#     "aklmnoi",
#     "kLpMn",
#     "LPM",
#     expect=Example("aBCDEFZYXxGVbcdefghklLPpMmnoi")
# )
# weights
test("abcde", "aBCDe", weights={"b": 10, "C": 10})
test("abcde", "aBCDe", weights={"c": 10, "B": 10})

test("SabcE", "SdefE", "SghiE", weights=[1, 7, 8, 9, 3, 6, 5, 7, 10, 9, 8])
test("_abcdefghij-", "_ABCDEFGHIJ-", "_klmnopqrst-", "_KLMNOPQRST-", expect=Example("_abcdefghijABCDEFGHIJklmnopqrstKLMNOPQRST-"))
# pathlen = 20
# npaths = 10
# chains = [chr(254) + "".join([chr(j) for j in range(pathlen * i, pathlen * (i + 1))]) + chr(255) for i in range(npaths)]
# test(*chains, "", "")
# test([chr(i) for i in range(3, 244)], "", "")


#test("SabcE", "SdefE", "SghiE", weights=[10, 7, 8, 9, 3, 6, 5, 7, 10, 9, 8])
test("SabcE", "SdefE", "SghiE", weights=[9, 5, 9, 4, 4, 10, 7, 10, 5, 8, 8])
test("abefst", "cde", "ghklmnopqrs", "ijk", weights=[3,2,6,3,7,0,3,2,3,2,6,3,6,2,5,3,5,3,3,2], expect=Example("ghijklmnabcdefopqrst"))

# Non-DAGs.
test("a", "A")
test("abc", "ABC")


tu.exit()
for i in range(0, 10):
    w = [random.randint(1, 10) for k in range(0, 11)]
    print("weights:", w)
    test("SabcE", "SdefE", "SghiE", weights=w)
