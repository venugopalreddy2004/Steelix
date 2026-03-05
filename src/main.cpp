#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include "onnx-ml.pb.h"
#include "ir.h"
#include "optimizer.h"

// IR is "Type-Agnostic" in its storage, but "Type-Aware" in its execution

int main() {
    onnx::ModelProto model;
    std::ifstream file("models/squeezenet_thick.onnx", std::ios::in | std::ios::binary);
    if(!model.ParseFromIstream(&file)) {
        std::cout << "Failed to load model" << std::endl;
        return 1;
    }

    onnx::GraphProto graph = model.graph();
    ir::Graph symTable;

    // we basically have 4 types of value nodes: initializer (ie weights and biases), input, output and intermediate 

    // this is initializer
    for(const auto& tensor : graph.initializer()) {
        ir::Value* val = symTable.create_value(tensor.name());
        val->shape.assign(tensor.dims().begin(), tensor.dims().end());
        val->is_init = true;
        
        // Detect Type
        val->type = (tensor.data_type() == onnx::TensorProto_DataType_INT64) ? 
                     ir::DType::INT64 : ir::DType::FLOAT32;

        // Copy raw bytes
        if(!tensor.raw_data().empty()) {
            val->raw_data.assign(tensor.raw_data().begin(), tensor.raw_data().end());
        }else if(val->type == ir::DType::FLOAT32 && tensor.float_data_size() > 0) {
            val->raw_data.resize(tensor.float_data_size() * sizeof(float));
            std::memcpy(val->raw_data.data(), tensor.float_data().data(), val->raw_data.size());
        }else if(val->type == ir::DType::INT64 && tensor.int64_data_size() > 0) {
            val->raw_data.resize(tensor.int64_data_size() * sizeof(int64_t));
            std::memcpy(val->raw_data.data(), tensor.int64_data().data(), val->raw_data.size());
        }
    }

    // input
    for(auto& ele : graph.input()) {
        if(symTable.value_map.find(ele.name()) == symTable.value_map.end()) {
            ir::Value* dataVal = symTable.create_value(ele.name());
            auto& t_type = ele.type().tensor_type();
            if(t_type.has_shape()) {
                dataVal->shape.clear();
                for (int i = 0; i < t_type.shape().dim_size(); i++)
                    dataVal->shape.push_back(t_type.shape().dim(i).dim_value());
            }
            symTable.modelIp.push_back(dataVal);
        }
    }

    // intermediates
    for(const auto& info : graph.value_info()) {
        ir::Value* v = symTable.create_value(info.name());
        auto& t_type = info.type().tensor_type();
        if(t_type.has_shape()) {
            v->shape.clear();
            for(int i = 0; i < t_type.shape().dim_size(); i++)
                v->shape.push_back(t_type.shape().dim(i).dim_value());
        }
    }

    // output
    for(const auto& out : graph.output()) {
        ir::Value* v = symTable.create_value(out.name());
        auto& t_type = out.type().tensor_type();
        if(t_type.has_shape()) {
            v->shape.clear();
            for(int i = 0; i < t_type.shape().dim_size(); i++)
                v->shape.push_back(t_type.shape().dim(i).dim_value());
        }
        symTable.modelOp.push_back(v);
    }

    // operator nodes
    for(auto& node : graph.node()) {
        if(node.op_type() == "Constant") {
            for(auto& attr : node.attribute()) {
                if(attr.name() == "value") {
                    auto& t = attr.t();
                    ir::Value* val = symTable.create_value(node.output(0));
                    val->is_init = true;
                    val->shape.assign(t.dims().begin(), t.dims().end());
                    val->raw_data.assign(t.raw_data().begin(), t.raw_data().end());
                }
            }
            continue;
        }

        std::string op_name = node.name().empty() ? node.output(0) : node.name();
        ir::Op* op = symTable.create_op(op_name, node.op_type());

        for(auto& in_name : node.input()) {
            ir::Value* in_val = symTable.create_value(in_name);
            op->inputs.push_back(in_val);
            in_val->consumers.push_back(op);
        }
        for(auto& out_name : node.output()) {
            ir::Value* out_val = symTable.create_value(out_name);
            op->outputs.push_back(out_val);
            out_val->producer = op;
        }
        for(auto& attr : node.attribute()) {
            if(attr.has_f()) op->float_attr[attr.name()] = attr.f();
            if(attr.ints_size() > 0) {
                for (int i = 0; i < attr.ints_size(); i++) 
                    op->int_attr[attr.name()].push_back(attr.ints(i));
            }else if(attr.has_i()) {
                op->int_attr[attr.name()] = { attr.i() };
            }
        }
    }

    symTable.toposort();

    std::set<ir::Value*> undying;
    for (const auto& onnx_out : graph.output()) {
        std::string name = onnx_out.name();
        if (symTable.value_map.count(name)) {
            undying.insert(symTable.value_map[name]);
        }
    }

    opt::PassManager pm;
    pm.addPass(std::make_unique<opt::identityCodeElim>());
    pm.addPass(std::make_unique<opt::deadCodeElim>());
    pm.addPass(std::make_unique<opt::constantFolding>());
    pm.converge(symTable, undying);

    onnx::ModelProto my_model = symTable.serialize();
    std::ofstream out("models/builded.onnx", std::ios::binary);
    my_model.SerializeToOstream(&out);
    out.close();
    return 0;
}