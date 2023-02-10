import ilpsolver
import functools
import math


class LayoutOpt:
    def __init__(self):
        self.ilp = ilpsolver.ILPSolver()
        self.s = self.ilp.s

    def solve(self, sz, conflicts):
        numBufs = len(sz)
        assert numBufs > 0

        # Scale all sizes down to GCD for highest accuracy downscaling.
        scaleFactor = functools.reduce(math.gcd, sz)
        sz = [s // scaleFactor for s in sz]

        # No number in the ILP may be larger than M, so scale down further if necessary.
        # The largest size is an estimate and not knowable without solving the problem.
        largestSize = 10 * max([sz[i] + sz[j] for i, j in conflicts])
        if largestSize >= self.ilp.M:
            squeezeFactor = math.ceil(largestSize / self.ilp.M)
            sz = [math.ceil(s / squeezeFactor) for s in sz]
            scaleFactor *= squeezeFactor
        elif largestSize * 100 <= self.ilp.M:
            # Gurobi has an issue when M is much larger than the largest size.
            self.ilp.M = largestSize * 100

        # e_i >= sz_i
        e = []
        for i in range(0, numBufs):
            e.append(self.s.IntVar(sz[i], self.s.Infinity(), "e_" + str(i)))

        # Buffer conflicts.
        for i, j in conflicts:
            self.makeNoOverlap(e[i], sz[i], e[j], sz[j])

        # max(endOffsets)
        maxE = self.s.IntVar(0, self.s.Infinity(), "maxE")
        for i in range(0, numBufs):
            self.ilp.greaterEqual(maxE, e[i])
        obj = self.ilp.getObjectiveFunc()
        obj.SetCoefficient(maxE, 1)

        self.ilp.solve()

        outOffsets = []
        for i in range(0, numBufs):
            outOffsets.append((int(e[i].solution_value()) - sz[i]) * scaleFactor)

        return outOffsets

    def makeNoOverlap(self, e_u, sz_u, e_v, sz_v):
        # no overlap if:
        # e_u - sz_u >= e_v  ||  e_v - sz_v >= e_u
        # rewritten as less equal:
        # e_v - e_u <= -sz_u
        # e_u - e_v <= -sz_v

        ct1, ct2 = self.ilp.makeOrConstraint(-sz_u, -sz_v)
        ct1.SetCoefficient(e_v, 1)
        ct1.SetCoefficient(e_u, -1)
        ct2.SetCoefficient(e_u, 1)
        ct2.SetCoefficient(e_v, -1)


def solve(sz, conflicts):
    return LayoutOpt().solve(sz, conflicts)
