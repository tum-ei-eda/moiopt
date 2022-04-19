import testutil as tu
import schedule


def test(*args, expect=None, weights={}):
    print("-----", *args, "-----")

    G = tu.makeGraph(*args, weights=weights)

    opt = schedule.SchedOpt()
    sched = opt.solve(G)
    sched = "".join(sched)

    if expect != None:
        tu.eq(sched, expect)
    else:
        tu.verifySched(sched, G)


# simple diamond
test("abc", "aBc")
# diamond in chain
test("abcde", "abCde", "abcCde")
# 2 parallel in chain
test("abcdef", "abCDef")
# 3 parallel
test("abc", "aBc", "aXc")
# consecutive
test("abcde", "aBcDe")
# nested
test("abcd", "aBc", "aXd")
test("abcd", "aBCd", "aBXd")
test("abcd", "aBCd", "aXCd")
# test("abcdme", "bCd", "aXMe")
# test("abcdme", "bCd", "aXMe", weights={"X": 1000})
# multi input
test("ab", "Ab")
test("abc", "ABc")
# test("abcd", "aBCd", "XYd")
test("abcd", "aBcd", "Xcd")
test("abcd", "aBcd", "Xd")
test("ab", "Ab", "XZb", "xZ")
# stress test
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
# test("abcdefghijk", "abcDEFg", "mnopqjk")
# test(
#     "abcdefghi",
#     "aBCDEFGh",
#     "CZYXVh",
#     "YxV",
#     "aklmnoi",
#     "kLpMn",
#     "LPM",
# )
# # weights
# test("abcde", "aBCDe", weights={"b": 10, "C": 10})
# test("abcde", "aBCDe", weights={"c": 10, "B": 10})

# test(
#     "SabcE",
#     "SdefE",
#     "SghiE",
#     weights={"S": 10, "a": 7, "b": 8, "c": 9, "d": 6, "e": 5, "f": 7, "g": 10, "h": 9, "i": 8, "E": 3},
# )
# test("SabcE", "SdefE", "SghiE", weights=[1, 7, 8, 9, 3, 6, 5, 7, 10, 9, 8])
# test("_abcdefghij-", "_ABCDEFGHIJ-", "_klmnopqrst-", "_KLMNOPQRST-", expect="_abcdefghij-")
# pathlen = 20
# npaths = 10
# chains = [chr(254) + "".join([chr(j) for j in range(pathlen * i, pathlen * (i + 1))]) + chr(255) for i in range(npaths)]
# test(*chains, "", "")
# test([chr(i) for i in range(3, 244)], "", "")
