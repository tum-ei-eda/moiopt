import testutil as tu
import pathdiscovery as pd


def makeNet(ops):
    net = []
    if "conv" in "".join(ops):
        expr = tu.relayOp("32x32x3", None)
    else:
        expr = tu.relayOp("100", None)
    net.append(expr)
    ops = ["dense10" if op == "dense" else op for op in ops]
    ops = ["conv10" if op == "conv" else op for op in ops]
    for op in ops:
        expr = tu.relayOp(op, expr)
        net.append(expr)
    return net


def test(ops, expectedSplitTypes, startOp=0, infer=None):
    if infer == None:
        infer = "d" * (len(ops) - startOp - 1)
    print("-----", ops, expectedSplitTypes, "-----")
    assert startOp < len(ops)
    assert set(infer) <= set("du")
    expectedSplitTypes = tu.translateSplitTypes(expectedSplitTypes)

    n = makeNet(ops)

    if pd.SplitType.FTP in expectedSplitTypes:
        testedNumPart = 4
    else:
        testedNumPart = 2
    testedStartType = expectedSplitTypes[min(len(expectedSplitTypes) - 1, infer.count("u"))]

    startIndex = endIndex = startOp + 1
    cfgs = pd.createAllSplitConfigs(n[startIndex])
    testedCfg = None
    for cfg in cfgs:
        if cfg.getNumPartitions() == testedNumPart and cfg.splitType == testedStartType:
            testedCfg = cfg
    status = testedCfg.inferUp()
    assert status == pd.InferStatus.VALID
    path = pd.SplitPath(testedCfg)
    for direction in infer:
        isUpwards = direction == "u"
        if isUpwards:
            currentIndex = startIndex = startIndex - 1
        else:
            currentIndex = endIndex = endIndex + 1
        newCfgs = pd.createChainedSplitConfigs(n[currentIndex], path, isUpwards)
        wasAdded = False
        for cfg in newCfgs:
            if cfg != None:
                wasAdded = path.addCfg(cfg, isUpwards)
                assert wasAdded
                break
        if not wasAdded:
            if isUpwards:
                startIndex += 1
            else:
                endIndex -= 1
            break

    tu.verifyPath(path, expectedSplitTypes, testedNumPart)


# Simple combinations.
test(["add", "add"], "==")
test(["add", "relu"], "==")
test(["add", "dense"], "=>")
test(["relu", "add"], "==")
test(["relu", "relu"], "==")
test(["relu", "dense"], "=>")
test(["dense", "add"], "<=")
test(["dense", "relu"], "<=")
test(["dense", "dense"], "<>")

# Reshape
test(["add", "reshape"], "==")
test(["reshape", "add"], "==")
test(["relu", "reshape"], "==")
test(["reshape", "relu"], "==")
test(["dense", "reshape"], "<=")
test(["reshape", "dense"], "=>")

# Longer chains.
test(["dense", "add", "dense", "mult"], "<=>~")
test(["dense", "add", "relu", "add", "relu", "dense", "mult", "mult"], "<====>~~")
test(["dense", "add", "dense", "mult"], "<=>~", startOp=1, infer="udd")
test(["dense", "add", "dense", "add"], "<=", startOp=2)

# Invalid chains.
test(["dense", "concat"], "<")
test(["add", "concat"], "=")
test(["relu", "concat"], "=")
test(["dense", "dense", "add"], "<", startOp=1, infer="u")
test(["dense", "dense", "relu"], "<>")
test(["dense", "dense", "dense"], "<>")
test(["dense", "dense", "mult", "relu"], "<>~")
test(["dense", "dense", "mult", "dense"], "<>~")

# Batched dense.
test(["dense10b5", "add"], "<=")
test(["dense10b5", "relu"], "<=")
test(["dense10b5", "dense"], "<>")
test(["dense10b5", "dense10b5"], "<>")
test(["add", "dense10b5"], "=>")
test(["relu", "dense10b5"], "=>")

# Combinations involving conv.
test(["add", "conv"], "=>")
test(["relu", "conv"], "=>")
test(["conv", "add"], "<=")
test(["conv", "relu"], "<=")
test(["conv", "conv"], "<>")
test(["conv", "conv10dw"], "<=")
test(["conv", "pool"], "<=")
test(["conv10dw", "conv"], "=>")
test(["pool", "conv"], "=>")
# test(["conv", "flatten", "dense"], "<=>")
# test(["conv", "flatten", "dense10b5"], "<=>")
test(["conv", "conv", "conv10dw"], "<>~")
test(["conv", "add", "relu", "pool", "conv10dw", "relu", "conv", "mult", "conv10dw", "mult"], "<=====>~~~")
test(["convpad", "convpad"], "<>")
test(["conv", "reshape"], "<=")
test(["reshape", "conv"], "=>")
test(["pad", "conv"], "=>")
test(["conv", "pad"], "<=")
test(["convstride2", "add"], "<=")

# FTP.
test(["add", "conv"], "##")
test(["relu", "conv"], "##")
test(["conv", "add"], "##")
test(["conv", "relu"], "##")
test(["conv", "conv"], "##")
test(["conv", "conv10dw"], "##")
test(["conv", "pool"], "##")
test(["conv10dw", "conv"], "##")
test(["pool", "conv"], "##")
test(["pool", "pool", "conv"], "###")
test(["conv", "pool", "pool"], "###")
test(["pool", "pool", "conv"], "###", startOp=1, infer="ud")
test(["conv", "pool", "pool"], "###", startOp=1, infer="ud")
test(["conv", "pool", "pool"], "###", startOp=2, infer="uu")
test(["conv", "pool", "pool", "pool"], "####", startOp=3, infer="uuu")
test(["convpad", "convpad"], "##")
test(["conv", "add", "add"], "###")
test(["conv", "conv", "conv"], "###")
test(["conv", "conv", "conv", "conv", "conv"], "#####")
test(["conv", "concat"], "#")
test(["conv", "flatten"], "#")  # TODO: in principle this is possible, but too complex for now

test(["cast64", "add", "add"], "===")
test(
    [
        "cast8",
        "conv",
        "add",
        "add",
        "cast64",
        "add",
        "add",
        "add",
        "cast32",
        "add",
        "add",
        "cast8",
        "add",
        "add",
        "conv",
    ],
    "<============>",
    startOp=1,
)

test(["conv20", "pool", "conv40", "pool", "conv80", "pool"], "######")

test(["dense5", "add", "relu", "reshape", "dense1", "relu"], "<===>")
test(["dense200b8", "add", "relu", "reshape", "dense100b5", "relu"], "<===>", startOp=1, infer="udddd")

test(["add", "relu", "pool", "convpad"], "####", startOp=1, infer="udd")
test(["cast8", "add", "pad", "conv20dwstride2", "add", "cast64"], "######", startOp=4, infer="uuuud")
test(
    [
        "conv8stride2",
        "cast8",
        "pad",
        "conv8dw",
        "cast8",
        "conv16k1",
        "cast8",
        "pad1",
        "conv16dwstride2",
        "cast8",
        "conv32k1",
        "cast8",
        "pad",
        "conv32dw",
    ],
    "##############",
)
test(
    [
        "pool",
        "pad",
        "conv128",
        "cast8",
        "pad",
        "conv128",
        "cast8",
        "pool",
        "pad",
        "conv256",
        "cast8",
        "pad",
        "conv256",
        "cast8",
        "pad",
        "conv256",
        "cast8",
        "pool",
    ],
    "##################",
)
test(
    [
        "conv32stride2",
        "conv32",
        "pad",
        "conv64",
        "pool3stride2",
        "conv80k1",
        "conv192",
    ],
    "#######",
)
