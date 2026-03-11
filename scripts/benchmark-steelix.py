import torch
import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper
import sys, os, time
import matplotlib.pyplot as plt

# 1. SETUP PATHS
sys.path.append(os.path.abspath("build"))
import runner

def load_weights_steelix(model_path):
    model = onnx.load(model_path)
    weights = {}
    for init in model.graph.initializer:
        w_np = numpy_helper.to_array(init)
        weights[init.name] = torch.from_numpy(w_np).cuda().to(torch.float32)
    return weights

def measure_latency(func, input_data, iterations=100, warmup=25):
    # Warmup: Let Triton JIT compile and autotune
    for _ in range(warmup):
        _ = func(input_data)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        _ = func(input_data)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iterations

def run_benchmarks():
    # Use the dynamic model we created during verification
    model_path = "models/squeezenet_dynamic.onnx"
    if not os.path.exists(model_path):
        print("ERROR: models/squeezenet_dynamic.onnx not found. Run verify-steelix.py first!")
        return

    weights = load_weights_steelix(model_path)
    
    # Setup ORT Session
    providers = [('CUDAExecutionProvider', {'device_id': 0})]
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    batch_sizes = [1, 8, 16, 32]
    ort_latencies = []
    steelix_latencies = []

    print(f"\n{'Batch':<6} | {'ORT Latency':<15} | {'Steelix Latency':<15} | {'Speedup'}")
    print("-" * 55)

    for b in batch_sizes:
        # Initialize Steelix engine for this batch
        engine = runner.SteelixRuntime(weights, batch_size=b)
        
        # Prepare inputs
        x_cpu = np.random.randn(b, 3, 224, 224).astype(np.float32)
        x_torch = torch.from_numpy(x_cpu).cuda().to(torch.float16)

        # Benchmark ORT
        ort_func = lambda x: sess.run(None, {input_name: x})
        t_ort = measure_latency(ort_func, x_cpu)
        ort_latencies.append(t_ort)

        # Benchmark Steelix
        stlx_func = lambda x: engine.forward(x)
        t_stlx = measure_latency(stlx_func, x_torch)
        steelix_latencies.append(t_stlx)

        speedup = t_ort / t_stlx
        print(f"{b:<6} | {t_ort:>10.4f} ms | {t_stlx:>10.4f} ms | {speedup:>8.2f}x")

    return batch_sizes, ort_latencies, steelix_latencies

def generate_report(batches, ort_data, stlx_data):
    plt.figure(figsize=(10, 6))
    
    # Calculate Speedups
    speedups = [ort / stlx for ort, stlx in zip(ort_data, stlx_data)]
    
    # Create Bar Chart
    x = np.arange(len(batches))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Latency Bars
    rects1 = ax1.bar(x - width/2, ort_data, width, label='ONNX Runtime (cuDNN)', color='#e74c3c')
    rects2 = ax1.bar(x + width/2, stlx_data, width, label='Steelix-AOT (Triton Fused)', color='#2ecc71')

    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Steelix-AOT vs ONNX Runtime: End-to-End Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batches)
    ax1.legend()

    # Add Speedup Labels on top
    for i, speedup in enumerate(speedups):
        ax1.text(i, max(ort_data[i], stlx_data[i]) + 0.1, f"{speedup:.2f}x Speedup", 
                 ha='center', fontweight='bold', color='#27ae60')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/performance-results.png')
    print("\nPerformance chart saved as 'performance_results.png'")

if __name__ == "__main__":
    batches, ort_res, stlx_res = run_benchmarks()
    generate_report(batches, ort_res, stlx_res)