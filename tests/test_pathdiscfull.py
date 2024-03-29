import testutil as tu


def test(ops, expectedNumPart, expectedSplitTypes=""):
    print("-----", ops, expectedNumPart, expectedSplitTypes, "-----")
    expectedSplitTypes = tu.translateSplitTypes(expectedSplitTypes)
    mod, splitPath = tu.opsStrToModAndSplitPath(ops)
    print(splitPath)
    if splitPath == None:
        tu.eq(expectedNumPart, 1)
    else:
        tu.verifyPath(splitPath, expectedSplitTypes, expectedNumPart)


test(["100", "dense1000", "add", "dense10"], 9, "<=>")
test(["100", "dense150", "add", "dense20"], 3, "<=>")
test(["100", "dense200", "add", "dense30"], 3, "<=>")
test(["100", "dense200", "add", "dense100"], 1)
test(["100", "dense150", "add", "dense100"], 1)
test(["100", "dense300", "add", "dense100"], 2, "<=>")
test(["100", "dense400", "add", "dense100"], 2, "<=>")
test(["100", "dense500", "add", "dense100"], 2, "<=>")
test(["100", "dense500", "reshape", "add", "dense100"], 3, "<==>")
test(["100", "dense200", "cast8", "dense100"], 1)
test(["100", "dense100", "cast8", "dense100"], 1)

test(["32x32x3", "conv20", "add", "conv10"], 3, "###")
test(["32x32x3", "conv20pad", "add", "conv10pad"], 3, "###")
test(["32x32x3", "pad", "conv20", "pad", "conv5"], 25, "####")
test(["32x32x3", "conv200", "add", "conv10"], 25, "###")
test(["32x32x3", "conv2000", "add", "conv10"], 25, "###")
test(["32x32x3", "conv1000", "add", "conv10"], 25, "###")
test(["32x32x3", "conv20stride2", "add", "conv5stride2"], 25, "###")

test(["32x32x3", "conv20", "conv", "pool"], 25, "###")
test(["32x32x3", "conv20", "conv40", "conv80", "pool", "conv100", "pool", "conv120", "pool"], 25, "######")

test(["1", "dense5", "add", "relu", "reshape", "dense1", "relu"], 4, "<===>")
test(["100", "dense200b8", "add", "relu", "reshape", "dense100b5", "relu"], 2, "<===>")
test(["100x100x3", "conv10pad", "add", "relu", "pool", "conv10pad", "add", "relu", "pool"], 25, "########")

test(
    [
        "96x96x3",
        "cast8",
        "pad1",
        "97x97x3",
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
        "cast8",
        "conv32k1",
        "cast8",
        "pad1",
        "conv32dwstride2",
        "cast8",
        "conv64k1",
        "cast8",
        "pad",
        "conv64dw",
        "cast8",
        "conv64k1",
        "cast8",
        "pad1",
        "conv64dwstride2",
        "cast8",
        "conv128k1",
        "cast8",
    ],
    1,
    "##########",
)
test(
    [
        "224x224x3",
        "cast8",
        "pad",
        "conv64",
        "cast8",
        "pad",
        "conv64",
        "cast8",
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
        "pad",
        "conv512",
        "cast8",
    ],
    25,
    "###########################",
)
test(
    [
        "299x299x3",
        "conv32stride2",
        "conv32",
        "pad",
        "conv64",
        "pool3stride2",
        "conv80k1",
        "conv192",
        "pool3stride2",
    ],
    25,
    "########",
)
test(
    [
        "96x96x3i8",
        "pad1",
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
        "cast8",
        "conv32k1",
        "cast8",
        "pad1",
        "conv32dwstride2",
        "cast8",
        "conv64k1",
        "cast8",
        "pad",
        "conv64dw",
        "cast8",
        "conv64k1",
        "cast8",
        "pad1",
        "conv64dwstride2",
        "cast8",
    ], 3, "<===="
)
test(["12x12x5", "conv15", "conv2", "pool", "flatten", "dense10"], 16, "###")
test(["32x8x1", "conv32k6rect2stride2pad22", "convdwpad", "conv64k1", "addbc64", "relu", "pool16rect4"], 3, "<===")

test(["100", "dense10000", "dense10", ("dense1000", 0), "dense10", ("cat", [2, 4])], 12, "<>")
test(["100", "dense10000", "dense10", ("dense1000", 0), "dense10", ("dense1000", 0), "dense10", ("cat", [2, 4, 6])], 12, "<>")
test(["100", "dense1000", ("dense1000", 0), ("cat", [1, 2]), "dense10000", "dense10", ("dense1000", 3), "dense10", ("cat", [5, 7])], 12, "<>")
test(["100", "dense1000", ("dense1000", 0), ("mult", [1, 2]), "dense10"], 1)

# test(["100", "flatten", "dense10"], 3, "=>")
# test(["2x1000", "reshapestrip", "dense10"], 9, "=>")
# test(["2x1000", "add", "reshapestrip", "dense10"], 9, "==>")
# test(["2x100", "add", "reshapestrip", "nndense1000", "dense10"], 9, "<>")
# test(["2x100", "add", "reshapestrip", "nndense1000", "transpose", "transpose", "dense10"], 12, "<==>")
