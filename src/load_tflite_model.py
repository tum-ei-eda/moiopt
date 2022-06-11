import tflite
from tflite.TensorType import TensorType as TType
from tvm import relay


class TensorInfo:
    def __init__(self, t, fix_names=False):
        self.name = t.Name().decode()
        if fix_names:
            self.name = self.name.replace("/", "_").replace(";", "_")

        typeLookup = {
            TType.FLOAT32: (4, "float32"),
            TType.UINT8: (1, "uint8"),
            TType.INT8: (1, "int8"),
            TType.INT16: (2, "int16"),
            TType.INT32: (4, "int32"),
            TType.INT64: (8, "int64"),
        }
        self.tysz, self.ty = typeLookup[t.Type()]
        assert self.ty != ""

        shape = tuple([t.Shape(si) for si in range(0, t.ShapeLength())])
        self.shape = shape

        self.size = self.tysz
        for dimSz in self.shape:
            self.size *= dimSz


class ModelInfo:
    def __init__(self, model, fix_names=False):
        assert model.SubgraphsLength() == 1
        g = model.Subgraphs(0)

        self.inTensors = []
        for i in range(0, g.InputsLength()):
            t = g.Tensors(g.Inputs(i))
            self.inTensors.append(TensorInfo(t, fix_names=fix_names))

        self.outTensors = []
        for i in range(0, g.OutputsLength()):
            t = g.Tensors(g.Outputs(i))
            self.outTensors.append(TensorInfo(t, fix_names=fix_names))


# Returns the TVM mod, TVM params and ModelInfo.
def load_tflite_model(modelBuf):
    tflModel = tflite.Model.GetRootAsModel(modelBuf, 0)

    shapes = {}
    types = {}

    modelInfo = ModelInfo(tflModel)
    for t in modelInfo.inTensors:
        shapes[t.name] = t.shape
        types[t.name] = t.ty
    for t in modelInfo.outTensors:
        shapes[t.name] = t.shape
        types[t.name] = t.ty

    mod, params = relay.frontend.from_tflite(tflModel, shape_dict=shapes, dtype_dict=types)
    return mod, params, modelInfo
