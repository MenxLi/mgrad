#include "nn.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <string>

namespace nn {

// create a new leaf node
Node& Graph::create_var(fp_t value, std::string name) {
    Node* node = new Node(this, name, value);
    nodes.push_back(node);
    return *node;
}
Node& Graph::create_const(fp_t value, std::string name) {
    Node& node = create_var(value, name);
    node.requires_grad = false;
    return node;
}

#define IMPL_GRAPH_OP1(OP) \
    Op##OP* op = new Op##OP(); \
    Node* node = new Node(this); \
    node->op = op; \
    op->inputs = {a}; \
    op->output = node; \
    ops.push_back(op); \
    nodes.push_back(node); \
    return node; \

#define IMPL_GRAPH_OP2(OP) \
    Op##OP* op = new Op##OP(); \
    Node* node = new Node(this); \
    node->op = op; \
    op->inputs = {a, b}; \
    op->output = node; \
    ops.push_back(op); \
    nodes.push_back(node); \
    return node; \

Node* Graph::add(Node* a, Node* b) { IMPL_GRAPH_OP2(Add) }
Node* Graph::sub(Node* a, Node *b) { IMPL_GRAPH_OP2(Sub) }
Node* Graph::mul(Node* a, Node* b) { IMPL_GRAPH_OP2(Mult) }
Node* Graph::div(Node* a, Node* b) { IMPL_GRAPH_OP2(Div) }
Node* Graph::pow(Node* a, Node* b) { IMPL_GRAPH_OP2(Pow) }
Node* Graph::log(Node* a) { IMPL_GRAPH_OP1(Log) }
Node* Graph::minus(Node* a) { IMPL_GRAPH_OP1(Minus) }
Node* Graph::inv(Node* a) { IMPL_GRAPH_OP1(Inv) }
Node* Graph::abs(Node* a) { IMPL_GRAPH_OP1(Abs) }
Node* Graph::relu(Node* a) { IMPL_GRAPH_OP1(Relu) }
Node* Graph::sigmoid(Node* a) { IMPL_GRAPH_OP1(Sigmoid) }
Node* Graph::tanh(Node* a) { IMPL_GRAPH_OP1(Tanh) }
Node* Graph::sin(Node* a) { IMPL_GRAPH_OP1(Sin) }
Node* Graph::cos(Node* a) { IMPL_GRAPH_OP1(Cos) }

Graph::~Graph() {
    for (Node* node: nodes) { delete node; }
    for (OpNode* op: ops) { delete op; }
}

void Graph::clear_grad() { for (Node* node: nodes) { node->grad = 0; } }
void Graph::forward() { for (OpNode* op: ops) { op->forward(); } }
void Graph::backward(Node* node) {
    assert(node->op != nullptr);
    node->grad = 1;
    for (int i = ops.size() - 1; i >= 0; i--) {
        ops[i]->backward();
    }
}


std::string node_id(Node* node) { return std::to_string((size_t)node); }
std::string node_id(OpNode* op) { return std::to_string((size_t)op); }
std::string Graph::to_mermaid() {
    std::string t = "graph TD;\n";
    auto create_node = [&t](Node* node) {
        std::stringstream ss;
        if (node->name != "") {
            ss << node->name << "@";
        }
        ss << "val: " << node->value;
        if (node->grad != 0) {
            assert(node->requires_grad);
            ss << ", grad: " << node->grad;
        }
        else if (!node->requires_grad){
            ss << ", const";
        }
        auto nid = node_id(node);
        t += nid+ "[" + ss.str() + "]\n";
        return nid;
    };
    auto create_op_node = [&t](OpNode* op) {
        auto nid = node_id(op);
        t += nid + "([" + op->name + "])\n";
        return nid;
    };
    
    for (Node* node: nodes) { create_node(node); }
    for (OpNode* op: ops) {
        auto op_id = create_op_node(op);
        for (Node* input: op->inputs) {
            t += node_id(input) + " --> " + op_id + "\n";
        }
        t += op_id + " --> " + node_id(op->output) + "\n";
    }
    return t;
}
}