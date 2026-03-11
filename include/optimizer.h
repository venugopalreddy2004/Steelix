#pragma once
#include "ir.h"
#include "onnx-ml.pb.h"
#include <vector>
#include <map>
#include <memory>
#include <unordered_set>
#include <set>
#include <stack>

namespace opt {
    class Pass{
        public:
            virtual ~Pass() = default;
            virtual std::string name() const = 0;

            virtual bool run(ir::Graph& graph, const std::set<ir::Value*>& immortals) = 0;
    };

    bool is_identity(ir::Op* targetOp, ir::Graph& symTable, const std::set<ir::Value*>& undying)
    {
        /*
            we will be checking if a given op node can be treated as an identity for our identity elimination

            conditions to check:
            1. does the node have more than 1 input and output values
            2. is output[0] global output
            3. if output has more than one members the consumers of others rather than output[0] should be 0
            4. type checking: {dropout, identity} => always safe , {reshape, squeeze, unsqueeze}=> conditionally safe; check thier ip & op type & shapes
        */

        if(targetOp->inputs.empty() || targetOp->outputs.empty()) return false;

        // if(std::find(symTable.modelOp.begin(), symTable.modelOp.end(), targetOp->outputs[0]) != symTable.modelOp.end()) return false;

        if(undying.count(targetOp->outputs[0])) return false;

        if(targetOp->outputs.size()>1){
            for(size_t i=1;i<targetOp->outputs.size();i++){
                if(targetOp->outputs[i]->consumers.size()) return false;
            }
        }
        
        auto& type = targetOp->type;
        if(type=="Dropout" || type=="Identity"){
            return true;
        }else if(type=="Reshape" || type=="Squeeze" || type=="Unsqueeze"){
            return (targetOp->inputs[0]->shape == targetOp->outputs[0]->shape && 
                targetOp->inputs[0]->type == targetOp->outputs[0]->type);
        }
        return false;   
    }

    class deadCodeElim : public Pass
    {
        public:
            std::string name() const override {return "Dead code elimination";}

            bool run(ir::Graph& graph, const std::set<ir::Value*>& undying)
            {
                /*
                    create 2 set containing of op and val which are alive
                    1. an op is alive if its outputs are dead values
                    2. a value is dead if it has 0 consumers and isnt a global model op

                    steps to implement:
                    1. create a set to hold alive values and ops
                    2. create a stack and insert all graph.input as they are always alive
                    3. while stack isnt empty pop value from it and insert in alive_val and 
                    insert the producer op of the value in alive_op. Also add the vals that are input to the stack
                    4. erase all ops that arent in the alive_ops, while doing this remove the op from consumer list of all its producers
                    5. erase all val that arent in the alive_val, while doing this remove that value from symTable.vals and symTable.value_map
                */

                std::unordered_set<ir::Value*> alive_val; std::unordered_set<ir::Op*> alive_op;

                size_t init_op_size = graph.ops.size(), init_val_size = graph.vals.size();

                std::stack<ir::Value*> all;
                for(auto& val : undying) all.push(val);

                while(!all.empty()){
                    ir::Value* val = all.top(); all.pop();
                    if(alive_val.count(val)) continue;
                    alive_val.insert(val);
                    ir::Op* op = val->producer;
                    if(op){
                        alive_op.insert(op);
                        for(ir::Value* ipVal : op->inputs){
                            if(alive_val.count(ipVal)==0) all.push(ipVal);
                        }
                    }
                }

                graph.ops.erase(std::remove_if(graph.ops.begin(), graph.ops.end(), [&](const std::unique_ptr<ir::Op>& op){
                    if(!alive_op.count(op.get())){
                        for(const auto& ipVal : op->inputs){
                            auto& c = ipVal->consumers;
                            c.erase(remove(c.begin(), c.end(), op.get()), c.end());
                        }
                        return true;
                    }
                    return false;
                }), 
                graph.ops.end());


                graph.vals.erase(std::remove_if(graph.vals.begin(), graph.vals.end(), [&](const std::unique_ptr<ir::Value>& val){
                    if(!alive_val.count(val.get())){
                        graph.value_map.erase(val->name);
                        return true;
                    }
                    return false; 
                }), 
                graph.vals.end());

                bool changed = ((graph.ops.size()!=init_op_size) || (graph.vals.size()!=init_val_size));
                return changed;
            }
    };

    class identityCodeElim : public Pass
    {
        public:
            std::string name() const override {return "Identity code elimination";}

            bool run(ir::Graph& symTable, const std::set<ir::Value*>& undying)
            {
                /*
                    we will be eliminating nodes that are like identity as they are mostly used in training and are of no value in runtime.

                    steps to implement:
                    1. iterate over symTable.ops to identify identity nodes
                    2. two pointer inVal (op->inputs[0]) and outVal (op->outputs[0])
                    3. for all the consumers of outVal, remove outVal from their inputs and insert inVal + insert consumer to inVal->consumer
                    4. clear outVal consumers list
                */
                for(const auto& op : symTable.ops){
                    if(is_identity(op.get(), symTable, undying)){
                        ir::Op* idOp = op.get();
                        ir::Value *inVal = idOp->inputs[0], *outVal = idOp->outputs[0];
                        for(auto& consumer : outVal->consumers){
                            std::replace(consumer->inputs.begin(), consumer->inputs.end(), outVal, inVal);
                            inVal->consumers.push_back(consumer);
                        }
                        outVal->consumers.clear();
                        return true;
                    }
                }
                return false;
            }
    };

    int const_type(ir::Op* node, ir::Graph& graph){
        auto type = node->type;
        std::set<std::string> metadataFoldGrp = {"Reshape", "Squeeze", "Unsqueeze", "Flatten", "Shape"};
        std::set<std::string> elemWiseFoldGrp = {"Add", "Mul", "Sub", "Div", "Relu", "Sqrt"};
        bool flag=true;
        for(const auto& ip : node->inputs){
            if(!ip->is_init){
                flag = (type=="Shape" && !ip->shape.empty()); break;
            }
        }

        if(flag && metadataFoldGrp.count(type)) return 1;
        else if(flag && elemWiseFoldGrp.count(type)) return 2;
        else return 0;
    }

    template <typename T, typename F>
    void unary_fold(const std::vector<ir::Value*>& input_vec, ir::Value* newVal, F op){
        const T* ip_ptr = reinterpret_cast<const T*>(input_vec[0]->raw_data.data());
        size_t num_elem = input_vec[0]->raw_data.size()/sizeof(T);

        newVal->raw_data.resize(num_elem * sizeof(T));
        T* out_ptr = reinterpret_cast<T*>(newVal->raw_data.data());
        for(size_t i=0;i<num_elem;i++){
            out_ptr[i] = op(ip_ptr[i]);
        }
    }

    template <typename T, typename F>
    void binary_fold(const std::vector<ir::Value*>& input_vec, ir::Value* newVal, F op){
        const T* ip_ptr1 = reinterpret_cast<const T*>(input_vec[0]->raw_data.data());
        const T* ip_ptr2 = reinterpret_cast<const T*>(input_vec[1]->raw_data.data());
        size_t num_elem = input_vec[0]->raw_data.size()/sizeof(T);

        newVal->raw_data.resize(num_elem * sizeof(T));
        T* out_ptr = reinterpret_cast<T*>(newVal->raw_data.data());
        for(size_t i=0;i<num_elem;i++){
            out_ptr[i] = op(ip_ptr1[i], ip_ptr2[i]);
        }
    }

    class constantFolding : public Pass{
        std::string name() const override {return "Constant Folding";}

        bool run(ir::Graph& graph, const std::set<ir::Value*>& immortals){
            for(const auto& node : graph.ops){
                 if (node->outputs[0]->consumers.empty() && immortals.count(node->outputs[0]) == 0) continue;

                int foldType = const_type(node.get(), graph);
                if(foldType){
                    if(foldType==1){
                        // metadata folding: only copy data and change metadata

                        /*
                            1. determine new shape
                            2. create new value, copy the raw_data to it and assign new shape
                            3. update output value wire's consumer list nodes (swap the old ip with new value)
                            4. clear old op value's consumer list
                        */

                        std::string type = node->type;
                        std::string original_name = node->outputs[0]->name;
                        ir::Value* newVal = graph.create_value(original_name + "_updated");
                        newVal->type = node->inputs[0]->type;
                        newVal->is_init = true;
                        newVal->producer=nullptr; 
                        //newVal->raw_data = node->inputs[0]->raw_data;

                        if(type=="Reshape"){
                            ir::Value* shape_info = node->inputs[1];
                            const int64_t * dims = reinterpret_cast<const int64_t*>(shape_info->raw_data.data());
                            size_t size = shape_info->raw_data.size()/sizeof(int64_t);

                            std::vector<int64_t> new_shape;
                            for(size_t i=0;i<size;i++){
                                new_shape.push_back(dims[i]);
                            }
                            newVal->shape=new_shape;
                            newVal->raw_data = node->inputs[0]->raw_data;
                        }else if(type=="Squeeze"){
                            std::set<int64_t> indices;
                            int64_t rank = node->inputs[0]->shape.size();
                            if(node->inputs.size()>1){
                                ir::Value* axes_info = node->inputs[1];
                                const int64_t* axes = reinterpret_cast<const int64_t*>(axes_info->raw_data.data());
                                size_t size = axes_info->raw_data.size()/sizeof(int64_t);

                                for(size_t i=0;i<size;i++){
                                    indices.insert((axes[i]<0)? axes[i]+rank : axes[i]);
                                }
                            }else if(node->int_attr["axes"].size()>0){
                                for(int64_t axis : node->int_attr["axes"]){
                                    indices.insert((axis<0)? (axis+rank) : (axis));
                                }
                            }else{
                                for(size_t i=0; i<node->inputs[0]->shape.size();i++){
                                    if(node->inputs[0]->shape[i]==1) indices.insert(i);  
                                }
                            }

                            // validity check
                            for (int64_t axis : indices) {
                                if (axis < 0 || axis >= rank) return false;
                                if (node->inputs[0]->shape[axis] != 1) return false;
                            }

                            std::vector<int64_t> new_shape;
                            for(size_t i=0;i<rank;i++){
                                if(indices.find(i)==indices.end()) new_shape.push_back(node->inputs[0]->shape[i]);
                            }
                            newVal->shape = new_shape;
                            newVal->raw_data = node->inputs[0]->raw_data;

                        }else if(type=="Unsqueeze"){
                            std::set<int64_t> indices;
                            int64_t rank = node->inputs[0]->shape.size(), targetRank = rank;

                            if(node->inputs.size()>1){
                                // modern systems
                                ir::Value *axes_info = node->inputs[1];
                                const int64_t *axes = reinterpret_cast<const int64_t*>(axes_info->raw_data.data());
                                size_t size =  axes_info->raw_data.size()/sizeof(int64_t);
                                targetRank+=(int64_t)size;

                                for(size_t i=0; i<size;i++){
                                    int64_t ai = ((axes[i]<0) ? (axes[i]+targetRank) : (axes[i]));
                                    if(ai<0 || ai>=targetRank) return false;
                                    indices.insert(ai);
                                }

                            }else{
                                // old ya legacy systems
                                const auto& axes = node->int_attr["axes"];
                                targetRank+=axes.size();
                                for(auto axis : axes){
                                    int64_t ai = ((axis<0) ? (axis+targetRank) : (axis));
                                    if(ai<0 || ai>=targetRank) return false;
                                    indices.insert(ai);
                                }
                            }

                            std::vector<int64_t> new_shape(targetRank,1);
                            size_t j=0;
                            for(size_t i=0;i<targetRank;i++){
                                if(indices.find(i)!=indices.end()) continue;
                                if(j<rank) new_shape[i]=node->inputs[0]->shape[j++];
                            }

                            newVal->shape=new_shape;
                            newVal->raw_data = node->inputs[0]->raw_data;

                        }else if(type=="Flatten"){
                            int64_t rank = node->inputs[0]->shape.size(), axis=1; // PS: aaxis=1 is default for ONNX
                            if(node->int_attr["axis"].size()>0) axis=node->int_attr["axis"][0];
                            if(axis<0) axis+=rank;

                            std::vector<int64_t> new_shape = {1,1};
                            for(int i=0;i<rank;i++){
                                int64_t val = node->inputs[0]->shape[i];
                                (i<axis) ? new_shape[0]*=val : new_shape[1]*=val;
                            }

                            newVal->raw_data = node->inputs[0]->raw_data;
                            newVal->shape = new_shape;


                        }else if(type=="Shape"){
                            int64_t rank = node->inputs[0]->shape.size();
                            newVal->shape = {rank};
                            newVal->raw_data.resize(rank*sizeof(int64_t));
                            std::memcpy(newVal->raw_data.data(), node->inputs[0]->shape.data(), newVal->raw_data.size());
                            newVal->type = ir::DType::INT64;

                        }

                        // Handshake                        
                        for(auto& consumer : node->outputs[0]->consumers){
                            std::replace(consumer->inputs.begin(), consumer->inputs.end(), node->outputs[0], newVal);
                            newVal->consumers.push_back(consumer);
                        }

                        node->outputs[0]->consumers.clear();
                        newVal->name = original_name;
                        graph.value_map[original_name] = newVal;   

                    }else{
                        // element wise folding: we have to do data manipulation

                        /*
                            1. DType check: check if all input are of same Dtype
                            2. reinterpret the data to the ip's Dtype * & allocate memory for output
                            3. stride calculations (not needed for unary and binary)
                            4. compute the operation 
                            5. create a new ir::value tensor and copy output as vector<char> to valu->raw_data
                            6. perform handshake
                        */

                        const auto& input_vec = node->inputs;
                        int64_t rank = input_vec.size();
                        ir::DType ip_dtype = input_vec[0]->type;
                        
                        // DType safety check
                        for(auto& ip_val : input_vec){
                            if(ip_val->type != ip_dtype) return false;
                        }
                        // shape alignment test
                        if(input_vec.size()==2){
                            if(input_vec[0]->shape != input_vec[1]->shape) return false;
                        }

                        std::string original_name = node->outputs[0]->name;
                        std::string op_type = node->type;
                        ir::Value* newVal = graph.create_value(original_name + "_updated");
                        newVal->is_init = true;
                        newVal->producer = nullptr;
                        newVal->type = input_vec[0]->type;
                        newVal->shape = input_vec[0]->shape; // true for atleas unary and binary op folding


                        // using template code for each Dtype
                        switch (ip_dtype)
                        {
                        case ir::DType::FLOAT32 :{
                            if(input_vec.size()==1){
                                if(op_type=="Relu"){
                                    unary_fold<float>(input_vec, newVal, [](float a){ return std::max(a,0.0f); });
                                }else if(op_type=="Sqrt"){
                                    unary_fold<float>(input_vec, newVal, [](float a){ return std::sqrt(a); });
                                }
                            }
                            else if(input_vec.size()==2){
                                if(op_type=="Add"){
                                    binary_fold<float>(input_vec, newVal, [](float a, float b){ return a+b; });
                                }else if(op_type=="Sub"){
                                    binary_fold<float>(input_vec, newVal, [](float a, float b){ return a-b; });
                                }else if(op_type=="Mul"){
                                    binary_fold<float>(input_vec, newVal, [](float a, float b){ return a*b; });
                                }else if(op_type=="Div"){
                                    binary_fold<float>(input_vec, newVal, [](float a, float b){ return (b==0) ? (0) : (a/b); });
                                }
                            }
                            break;
                        }

                        case ir::DType::INT64:{
                            if(input_vec.size()==1){
                                if(op_type=="Relu"){
                                    unary_fold<int64_t>(input_vec, newVal, [](int64_t a){ return std::max(a,(int64_t)0); });
                                }else if(op_type=="Sqrt"){
                                    unary_fold<int64_t>(input_vec, newVal, [](int64_t a){ return std::sqrt(a); });
                                }
                            }
                            else if(input_vec.size()==2){
                                if(op_type=="Add"){
                                    binary_fold<int64_t>(input_vec, newVal, [](int64_t a, int64_t b){ return a+b; });
                                }else if(op_type=="Sub"){
                                    binary_fold<int64_t>(input_vec, newVal, [](int64_t a, int64_t b){ return a-b; });
                                }else if(op_type=="Mul"){
                                    binary_fold<int64_t>(input_vec, newVal, [](int64_t a, int64_t b){ return a*b; });
                                }else if(op_type=="Div"){
                                    binary_fold<int64_t>(input_vec, newVal, [](int64_t a, int64_t b){ return (b==0) ? (0) : (a/b); });
                                }
                            }
                            break;
                        }

                        default: return false;
                        }

                        // Handshake
                        for(auto& consumer : node->outputs[0]->consumers){
                            std::replace(consumer->inputs.begin(), consumer->inputs.end(), node->outputs[0], newVal);
                            newVal->consumers.push_back(consumer);
                        }

                        node->outputs[0]->consumers.clear();
                        newVal->name = original_name;
                        graph.value_map[original_name] = newVal; 

                    }
                    return true;
                }
            }
            return false;
        }
    };

class operatorFusion : public Pass {
public:
    std::string name() const override { return "Operator Fusion (Conv+Bias+ReLU)"; }

    bool run(ir::Graph& graph, const std::set<ir::Value*>& undying) override {
        for (auto& op_ptr : graph.ops) {
            ir::Op* op = op_ptr.get();
            if (op->type != "Conv") continue;

            ir::Op* conv_node = op;
            ir::Value* conv_out = conv_node->outputs[0];
            if (conv_out->consumers.size() != 1 || undying.count(conv_out)) continue;

            ir::Op* next_node = conv_out->consumers[0];
            ir::Op* bias_node = nullptr;
            ir::Op* relu_node = nullptr;

            if (next_node->type == "Add") {
                ir::Value* b_cand = (next_node->inputs[0] == conv_out) ? next_node->inputs[1] : next_node->inputs[0];
                if (b_cand->is_init) {
                    bias_node = next_node;
                    ir::Value* add_out = bias_node->outputs[0];
                    if (add_out->consumers.size() == 1 && !undying.count(add_out)) {
                        if (add_out->consumers[0]->type == "Relu") relu_node = add_out->consumers[0];
                    }
                }
            } else if (next_node->type == "Relu") {
                relu_node = next_node;
            }

            if (!bias_node && !relu_node) continue;

            ir::Op* tail_node = relu_node ? relu_node : bias_node;
            ir::Value* tail_out = tail_node->outputs[0];
            std::string original_final_name = tail_out->name;

            ir::Op* fusedOp = graph.create_op(conv_node->name + "_fused", "FusedConv");
            ir::Value* fusedOut = graph.create_value(original_final_name + "_fused_wire");
            fusedOut->producer = fusedOp;
            fusedOut->shape = tail_out->shape;
            fusedOut->type = tail_out->type;
            fusedOp->outputs.push_back(fusedOut); 

            // Input Migration with Triple Handshake
            for (auto* in : conv_node->inputs) {
                fusedOp->inputs.push_back(in);
                in->consumers.push_back(fusedOp);
            }
            if (bias_node) {
                ir::Value* b = (bias_node->inputs[0] == conv_out) ? bias_node->inputs[1] : bias_node->inputs[0];
                fusedOp->inputs.push_back(b);
                b->consumers.push_back(fusedOp);
            }

            fusedOp->int_attr = conv_node->int_attr;
            fusedOp->float_attr = conv_node->float_attr;
            if (relu_node) fusedOp->int_attr["activation_relu"] = {1};

            std::vector<ir::Op*> consumers = tail_out->consumers;
            for (auto* c : consumers) {
                std::replace(c->inputs.begin(), c->inputs.end(), tail_out, fusedOut);
                fusedOut->consumers.push_back(c);
            }

            fusedOut->name = original_final_name;
            graph.value_map[original_final_name] = fusedOut;
            tail_out->consumers.clear();
            conv_out->consumers.clear();
            if (bias_node) bias_node->outputs[0]->consumers.clear();

            return true; 
        }
        return false;
    }
};

    class PassManager{
        public:
            std::vector<std::unique_ptr<Pass>> pipeline;

            void addPass(std::unique_ptr<Pass> pass){
                pipeline.push_back(std::move(pass));
            }

            void converge(ir::Graph& graph, std::set<ir::Value*>& undying){
                bool is_changed=true;
                while(is_changed){
                    is_changed=false;
                    for(const auto& pass : pipeline){
                        if(pass->run(graph, undying)) {
                            std::cout << "[Pass] " << pass->name() << " modified the graph." << std::endl;
                            is_changed=true; 
                        }
                    }
                }
                graph.toposort();
            }
    };

}

