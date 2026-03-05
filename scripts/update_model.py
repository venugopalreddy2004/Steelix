import onnx
from onnx import shape_inference

model = onnx.load("models/squeezenet1.1-7.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "models/squeezenet_thick.onnx")