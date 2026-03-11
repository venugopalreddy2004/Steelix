#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <queue>
#include <iostream>
#include "onnx-ml.pb.h"

namespace ir {
    struct Op;
    struct Value;
    enum class DType { FLOAT32, FLOAT16, INT64, INT32 };

    struct Value{
        std::string name;
        std::vector<int64_t> shape;
        DType type = DType::FLOAT32;
        bool is_init = false;
        std::vector<char> raw_data; 
        Op* producer = nullptr;
        std::vector<Op*> consumers;
        std::vector<int64_t> get_strides() {
            int64_t rank = this->shape.size();
            if (rank == 0) return {}; 

            std::vector<int64_t> strides(rank);
            int64_t current_stride = 1;

            for (int i = rank - 1; i >= 0; i--) {
                strides[i] = current_stride;
                current_stride *= this->shape[i];
            }
            return strides;
        }
    };

    struct Op{
        std::string name;
        std::string type;
        std::vector<Value*> inputs;
        std::vector<Value*> outputs;
        std::map<std::string, std::vector<int64_t>> int_attr;
        std::map<std::string, float> float_attr;
    };

    class Graph{
    public:
        std::vector<std::unique_ptr<Op>> ops;
        std::vector<std::unique_ptr<Value>> vals;
        std::map<std::string, Value*> value_map;
        std::vector<Value*> modelIp;
        std::vector<Value*> modelOp;

        Value* create_value(const std::string& name){
            if (value_map.find(name) != value_map.end()) return value_map[name];
            auto val = std::make_unique<Value>();
            val->name = name;
            Value* ptr = val.get();
            vals.push_back(std::move(val));
            value_map[name] = ptr;
            return ptr;
        }

        Op* create_op(const std::string& name, const std::string& type){
            auto op = std::make_unique<Op>();
            op->name = name; op->type = type;
            Op* ptr = op.get();
            ops.push_back(std::move(op));
            return ptr;
        }

        void toposort(){
            std::map<Op*, int> indegree;
            std::queue<Op*> q;
            for(const auto& op : ops) {
                int deg = 0;
                for(auto* ip : op->inputs) {
                    if(ip->producer != nullptr) deg++;
                }
                indegree[op.get()] = deg;
                if(deg == 0) q.push(op.get());
            }
            std::map<Op*, std::unique_ptr<Op>> temp;
            for(auto& op : ops) temp[op.get()] = std::move(op);
            ops.clear();
            std::vector<std::unique_ptr<Op>> topo;
            while(!q.empty()) {
                auto node = q.front(); q.pop();
                topo.push_back(std::move(temp[node]));
                for(auto* adjVal : node->outputs) {
                    for(auto* adjOp : adjVal->consumers) {
                        indegree[adjOp]--;
                        if (indegree[adjOp] == 0) q.push(adjOp);
                    }
                }
            }
            ops = std::move(topo);
        }

        onnx::ModelProto serialize(){
            onnx::ModelProto model;
            model.set_ir_version(7);
            auto* opset = model.add_opset_import();
            opset->set_domain("");
            opset->set_version(11); 
            onnx::GraphProto* g = model.mutable_graph();

            // initializers (they have is_init as true)
            for(auto& v : vals) {
                if(v->is_init) {
                    auto* init = g->add_initializer();
                    init->set_name(v->name);
                    for(auto d : v->shape) init->add_dims(d);
                    
                    if(v->type == ir::DType::INT64) {
                        init->set_data_type(onnx::TensorProto_DataType_INT64);
                    }else{
                        init->set_data_type(onnx::TensorProto_DataType_FLOAT);
                    }
                    init->set_raw_data(v->raw_data.data(), v->raw_data.size());
                }
            }

            // inputs (they have no producer)
            for(auto& v : vals) {
                if(v->producer == nullptr) { 
                    auto* input = g->add_input();
                    input->set_name(v->name);
                    auto* type = input->mutable_type()->mutable_tensor_type();
                    type->set_elem_type(v->type == DType::INT64 ? 
                        onnx::TensorProto_DataType_INT64 : onnx::TensorProto_DataType_FLOAT);
                    auto* shape = type->mutable_shape();
                    for(auto d : v->shape) shape->add_dim()->set_dim_value(d);
                }
            }

            // intermediate (they have a producer and >0 consumers)
            for(auto& v : vals) {
                if(v->producer != nullptr && !v->consumers.empty()) {
                    auto* vi = g->add_value_info();
                    vi->set_name(v->name);
                    auto* type = vi->mutable_type()->mutable_tensor_type();
                    type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
                    auto* shape = type->mutable_shape();
                    for(auto d : v->shape) shape->add_dim()->set_dim_value(d);
                }
            }

            // outputs (they have no consumers and isnt initialized)
            for(auto& v : vals) {
                if(v->consumers.empty() && !v->is_init) {
                    auto* output = g->add_output();
                    output->set_name(v->name);
                    auto* type = output->mutable_type()->mutable_tensor_type();
                    type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
                    auto* shape = type->mutable_shape();
                    for(auto d : v->shape) shape->add_dim()->set_dim_value(d);
                }
            }

            // operators 
            for(auto& op : ops) {
                auto* n = g->add_node();
                n->set_name(op->name);
                n->set_op_type(op->type);
                for(auto* in : op->inputs) n->add_input(in->name);
                for(auto* out : op->outputs) n->add_output(out->name);
                for(auto const& [key, val] : op->int_attr) {
                    auto* attr = n->add_attribute();
                    attr->set_name(key);
                    if(key == "axis" || key == "keepdims" || key == "group") {
                        attr->set_type(onnx::AttributeProto_AttributeType_INT);
                        attr->set_i(val[0]);
                    }else {
                        attr->set_type(onnx::AttributeProto_AttributeType_INTS);
                        for(auto i : val) attr->add_ints(i);
                    }
                }
                for(auto const& [key, val] : op->float_attr) {
                    auto* attr = n->add_attribute();
                    attr->set_name(key);
                    attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
                    attr->set_f(val);
                }
            }

            return model;
        }
    };
}