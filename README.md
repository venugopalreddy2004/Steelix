
# Steelix: High-Performance AOT Compiler for AI Inference

**Steelix** is an Ahead-of-Time (AOT) compiler written in C++ that transforms standard ONNX computation graphs into a custom Intermediate Representation (IR) optimized for high-throughput GPU execution. 

Unlike general-purpose runtimes that rely on interpretive overhead, Steelix performs deep graph surgery and symbolic shape propagation to minimize memory bandwidth bottlenecks.

##  Technical Architecture

### 1. Intermediate Representation (IR)
*   **Bi-partite Graph Design:** Uses a custom SSA-inspired IR with bidirectional links between `Op` (Operators) and `Value` (Tensors). 
*   **Memory Management:** Implements `std::unique_ptr` ownership for lifecycle management with raw-pointer "handshaking" for $O(1)$ graph traversal.
*   **Type-Agnostic Storage:** Employs a byte-buffer strategy (`std::vector<char>`) for weight ingestion, allowing bit-perfect serialization of `FLOAT32`, `INT64`, and `FLOAT16` data.

### 2. Optimization Pass Manager
*   **Fixed-Point Iteration:** Features a modular pipeline orchestrator that executes transformation passes recursively until the graph converges to a mathematically optimal state.
*   **Surgical Passes:**
    *   **Dead Code Elimination (DCE):** Mark-and-sweep reachability analysis starting from model outputs to prune unused branches.
    *   **Identity Elimination:** Automated bypass surgery for "No-Op" patterns (Dropout, Identity, redundant Reshapes).
    *   **Constant Folding:** Implementation of a "Metadata-to-Data" bridge (Shape/Gather waterfall) and arithmetic pre-computation for static weights.

### 3. Frontend & Serialization
*   **Protobuf Ingestion:** Native parsing of ONNX models via Google Protobuf.
*   **Topological Scheduling:** Implements Kahn’s Algorithm to guarantee a valid execution order for the generated backend.
*   **Identity Transform:** Verified ability to perform a lossless "Round-Trip" (ONNX ➔ IR ➔ ONNX) with zero numerical drift.

---

## 🛠 Setup & Installation

### Prerequisites
* **C++ Compiler:** GCC 11+ or Clang 14+ (C++17 required)
* **Protobuf:** `libprotobuf-dev` and `protobuf-compiler`
* **Python:** 3.10+ with `numpy` and `onnxruntime`

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/aot-compiler.git
cd aot-compiler

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Download the Model
Steelix uses **SqueezeNet 1.1** as its primary optimization benchmark.
1. Download `squeezenet1.1-7.onnx` from the [ONNX Model Zoo](https://github.com/onnx/models/blob/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx).
2. Place the file into the `models/` directory.
3. **Thicken the model** (inject shapes):
   ```bash
   python3 scripts/shape_inference.py
   ```

### 3. Build and Execute
The Makefile includes a comprehensive testing pipeline:
```bash
make test
```
This command compiles the C++ engine, runs the optimization passes, serializes the optimized graph to `optimized.onnx`, and triggers the Python numerical verification suite.

---

##  Benchmark Results (SqueezeNet 1.1)

| Metric | Original Model | Steelix Optimized | Status |
| :--- | :--- | :--- | :--- |
| **Op Count** | 66 | 58 | **-12%** |
| **Value Count** | 120 | 110 | **-8%** |
| **Numerical Match** | 1.0 | 1.0 | **Bit-Perfect** |

---

## Roadmap
- [ ] **Operator Fusion:** Greedy epilogue fusion for `Conv -> Bias -> ReLU` sequences to eliminate HBM round-trips.
- [ ] **Triton Backend:** C++ emitter logic to generate JIT-compiled Triton Python kernels for fused operators.
- [ ] **Autotuning:** Empirical search for optimal tiling sizes ($BLOCK\_SIZE$) based on hardware SRAM and L2 cache constraints.

