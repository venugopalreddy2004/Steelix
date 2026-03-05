import onnxruntime as ort
import numpy as np

# Load both sessions
try:
    orig_sess = ort.InferenceSession("models/squeezenet1.1-7.onnx")
    mine_sess = ort.InferenceSession("models/builded.onnx")

    # Get the input name from the ORIGINAL model
    # (It might be 'data', 'data_0', or 'input')
    input_name_orig = orig_sess.get_inputs()[0].name
    input_name_mine = mine_sess.get_inputs()[0].name
    
    print(f"Original model expects: {input_name_orig}")
    print(f"Your model expects: {input_name_mine}")

    # Create random input
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Run original
    out_orig = orig_sess.run(None, {input_name_orig: x})[0]

    # Run yours
    out_mine = mine_sess.run(None, {input_name_mine: x})[0]

    # Numerical Match
    match = np.allclose(out_orig, out_mine, atol=1e-4)
    print(f"Numerical Match: {match}")
    
    if not match:
        print(f"Max Difference: {np.max(np.abs(out_orig - out_mine))}")

except Exception as e:
    print(f"Test Failed: {e}")