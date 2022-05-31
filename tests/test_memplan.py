import testutil as tu
import memplanner


def test(*args):
    print("-----", *args, "-----")
    expected = args[-1]
    net = tu.makeNet(*args[:-1])
    planner = memplanner.MemoryPlanner(net.createBestSchedule())
    memLayout = planner.createOptimalLayout()

    # Small error is okay for very large values that exceed the solvers upper bound.
    tu.almosteq(memLayout.getSize(), expected, delta=0.000002 * expected)


# Basic
test("abcdef", [1, 20, 6, 4, 8, 1], 26)
# Case where we are better than TFLM
test("abcd", [5, 3, 2, 4], 8)
# Parallel
test("abc", "aBc", [5, 10, 5, 12], 27)
# Performance
# test(
#     "abcdefghijklmnopqrstuvwxyz",
#     "aBCDEFGHIJKLMNOPQRSTUVWXYz",
#     "a1234567890!'§$%&/()=#+*~z",
#     [5] * 26 + [3] * 24 + [7] * 24,
#     22,
# )
# test("".join([chr(i) for i in range(256)]), [5] * 256, 10)
# Large values
test("abcd", [6000000, 6000000, 2000000, 1000000], 12000000)
test("abcd", [6000000, 6000000, 2000000, 1], 12000000)
test("abcd", [6000000, 6000001, 2000000, 1], 12000008)
