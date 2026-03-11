#pragma once
#include "ir.h"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

namespace backend {

class TritonEmitter {
private:
    const std::string KERNEL_TEMPLATE = R"(
@triton.autotune(configs=autotune_configs, key=['M','N','K'])
@triton.jit
def _fused_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_batch_a,
    stride_bk, stride_bn, stride_batch_b,
    stride_cm, stride_cn, stride_batch_c,
    ENABLE_RELU: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE: tl.constexpr    
):
    batch_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    a_ptr += batch_id * stride_batch_a
    b_ptr += batch_id * stride_batch_b
    c_ptr += batch_id * stride_batch_c

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    a_offsets = rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_offsets = rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_x = (rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_SIZE_K)
        mask_w = (rk[:, None] < K - k * BLOCK_SIZE_K) & (rn[None, :] < N)
        a = tl.load(a_ptr + a_offsets, mask=mask_x, other=0.0).to(tl.float16)
        b = tl.load(b_ptr + b_offsets, mask=mask_w, other=0.0).to(tl.float16)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32, allow_tf32=True)
        a_offsets += BLOCK_SIZE_K * stride_ak
        b_offsets += BLOCK_SIZE_K * stride_bk

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    if ENABLE_RELU:
        acc = tl.maximum(0.0, acc) 

    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr + c_offsets, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))
)";

    std::string sanitize(std::string name) {
        std::replace(name.begin(), name.end(), '/', '_');
        std::replace(name.begin(), name.end(), '.', '_');
        std::replace(name.begin(), name.end(), ':', '_');
        std::replace(name.begin(), name.end(), '-', '_');
        return "v_" + name;
    }

    std::string get_val(ir::Value* v) {
        if (v->is_init) return "weights['" + v->name + "']";
        return "self.vars['" + v->name + "']";
    }

public:
    void emit(ir::Graph& graph, const std::string& output_dir) {
        std::ofstream k_file(output_dir + "/generated_kernels.py");
        std::ofstream r_file(output_dir + "/runner.py");

        k_file << "import torch\nimport triton\nimport triton.language as tl\n\n";
        k_file << "autotune_configs = [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4)]\n";
        k_file << KERNEL_TEMPLATE << "\n";

        r_file << "import torch\nimport torch.nn.functional as F\nimport generated_kernels\nimport triton\n\n";
        r_file << "class SteelixRuntime:\n";
        r_file << "    def __init__(self, weights, batch_size=1):\n";
        r_file << "        self.weights = weights\n";
        r_file << "        self.batch_size = batch_size\n";
        r_file << "        self.vars = {}\n\n";
        
        r_file << "    def forward(self, input_tensor):\n";
        r_file << "        weights = self.weights\n";
        r_file << "        batch_size = self.batch_size\n";
        
        if (!graph.modelIp.empty()) {
            r_file << "        self.vars['" << graph.modelIp[0]->name << "'] = input_tensor.to(torch.float16).cuda()\n\n";
        }

        for (auto& op_ptr : graph.ops) {
            ir::Op* op = op_ptr.get();
            std::string out = op->outputs[0]->name;
            std::string in0 = get_val(op->inputs[0]);

            // --- 1. POOLING (MANDATORY FIX) ---
            if (op->type == "GlobalAveragePool" || op->type == "AveragePool") {
                r_file << "        self.vars['" << out << "'] = F.adaptive_avg_pool2d(" << in0 << ", (1, 1))\n\n";
                continue;
            }

            // --- 2. TRITON GEMM (1x1) ---
            bool is_1x1 = (op->int_attr.count("kernel_shape") && op->int_attr.at("kernel_shape")[0] == 1);
            if (op->type == "FusedConv" && is_1x1) {
                int64_t Ci = op->inputs[0]->shape[1], H = op->inputs[0]->shape[2], W = op->inputs[0]->shape[3];
                int64_t Co = op->inputs[1]->shape[0], M = H * W, K = Ci, N = Co;
                bool has_relu = op->int_attr.count("activation_relu");

                r_file << "        # " << op->name << " (Triton)\n";
                r_file << "        A = " << in0 << ".reshape(batch_size, " << K << ", " << M << ").transpose(1, 2).contiguous()\n";
                r_file << "        W = weights['" << op->inputs[1]->name << "'].reshape(" << Co << ", " << K << ").transpose(0, 1).contiguous().to(torch.float16)\n";
                r_file << "        C = torch.empty((batch_size, " << M << ", " << N << "), device='cuda', dtype=torch.float16)\n";
                r_file << "        grid = lambda meta: (triton.cdiv(" << M << ", meta['BLOCK_SIZE_M']) * triton.cdiv(" << N << ", meta['BLOCK_SIZE_N']), batch_size)\n";
                r_file << "        generated_kernels._fused_kernel[grid](A, W, C, weights['" << op->inputs[2]->name << "'].to(torch.float16).cuda(), "
                       << M << ", " << N << ", " << K << ", "
                       << "A.stride(1), A.stride(2), A.stride(0), W.stride(0), W.stride(1), 0, C.stride(1), C.stride(2), C.stride(0), "
                       << (has_relu ? "True" : "False") << ")\n";
                r_file << "        self.vars['" << out << "'] = C.transpose(1, 2).reshape(batch_size, " << Co << ", " << H << ", " << W << ")\n\n";
            }
            // --- 3. CONV FALLBACK ---
            else if (op->type == "Conv" || op->type == "FusedConv") {
                r_file << "        res = F.conv2d(" << in0 << ".to(torch.float32), weights['" << op->inputs[1]->name << "'].to(torch.float32), "
                       << (op->inputs.size() > 2 ? "weights['" + op->inputs[2]->name + "'].to(torch.float32)" : "None") << ", "
                       << "stride=" << (op->int_attr.count("strides") ? std::to_string(op->int_attr.at("strides")[0]) : "1") << ", "
                       << "padding=" << (op->int_attr.count("pads") ? std::to_string(op->int_attr.at("pads")[0]) : "0") << ").to(torch.float16)\n";
                if (op->int_attr.count("activation_relu") || op->type == "Relu") r_file << "        res = F.relu(res)\n";
                r_file << "        self.vars['" << out << "'] = res\n\n";
            }
            // --- 4. ACTIVATION & SOFTMAX ---
            else if (op->type == "Relu") {
                r_file << "        self.vars['" << out << "'] = F.relu(" << in0 << ")\n\n";
            }
            else if (op->type == "Softmax") {
                r_file << "        self.vars['" << out << "'] = F.softmax(" << in0 << ".reshape(batch_size, -1), dim=1)\n\n";
            }
            // --- 5. OTHERS (Concat, MaxPool) ---
            else if (op->type == "MaxPool") {
                r_file << "        self.vars['" << out << "'] = F.max_pool2d(" << in0 << ", kernel_size=" << op->int_attr.at("kernel_shape")[0] << ", stride=" << op->int_attr.at("strides")[0] << ")\n\n";
            }
            else if (op->type == "Concat") {
                r_file << "        self.vars['" << out << "'] = torch.cat([";
                for(size_t i=0; i<op->inputs.size(); i++) { r_file << get_val(op->inputs[i]) << (i == op->inputs.size()-1 ? "" : ", "); }
                r_file << "], dim=1)\n\n";
            }
            else {
                r_file << "        self.vars['" << out << "'] = " << in0 << ".reshape(batch_size, -1)\n\n";
            }
        }
        if (!graph.modelOp.empty()) r_file << "        return self.vars['" << graph.modelOp[0]->name << "']\n";
    }
};

}