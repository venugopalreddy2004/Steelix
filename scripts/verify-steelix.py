import sys, os, torch, onnx, numpy as np, onnxruntime as ort
from onnx import numpy_helper

sys.path.append(os.path.abspath("build"))
import runner

def load_weights_steelix(model_path):
    model = onnx.load(model_path)
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = torch.from_numpy(numpy_helper.to_array(init)).cuda().to(torch.float32)
    return weights

def run_test(batch_size=1):
    model_path = "models/squeezenet1.1-7.onnx" if batch_size==1 else "models/squeezenet_dynamic.onnx"
    weights = load_weights_steelix(model_path)
    engine = runner.SteelixRuntime(weights, batch_size=batch_size)
    
    torch.manual_seed(42)
    x_cpu = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    x_torch = torch.from_numpy(x_cpu).cuda().to(torch.float16)

    print(f"Running Steelix-AOT (Batch {batch_size})...")
    _ = engine.forward(x_torch) # Warmup
    out_steelix = engine.forward(x_torch).flatten()

    print(f"Running ORT Reference...")
    # Use dynamic session for batch > 1
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    out_ort = sess.run(None, {sess.get_inputs()[0].name: x_cpu})[0]
    out_ort = torch.from_numpy(out_ort).cuda().to(torch.float16).flatten()

    match = torch.allclose(out_steelix, out_ort, atol=2e-2, rtol=2e-2)
    print(f"NUMERICAL MATCH: {match}")
    if not match:
        print(f"Steelix Size: {out_steelix.shape}, ORT Size: {out_ort.shape}")
        print(f"Max Diff: {torch.max(torch.abs(out_steelix - out_ort))}")
        
def make_model_dynamic(input_path, output_path):
    model = onnx.load(input_path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    
    onnx.save(model, output_path)

if __name__ == "__main__":
    make_model_dynamic("models/squeezenet1.1-7.onnx", "models/squeezenet_dynamic.onnx")
    run_test(1)
    run_test(32)