#include "nn.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <string>
#include <iomanip>

namespace nn {

// create a new leaf node
Node* Graph::create_var(fp_t value, std::string name) {
    Node* node = new Node(this, name, value);
    nodes.push_back(node);
    return node;
}
Node* Graph::create_const(fp_t value, std::string name) {
    Node* node = create_var(value, name);
    node->requires_grad = false;
    return node;
}

#define IMPL_GRAPH_OP(OP, ...) \
    Op##OP* op = new Op##OP(); \
    Node* node = new Node(this); \
    node->op = op; \
    op->inputs = {__VA_ARGS__}; \
    op->output = node; \
    ops.push_back(op); \
    nodes.push_back(node); \
    return node; \

Node* Graph::add(Node* a, Node* b) { IMPL_GRAPH_OP(Add, a, b) }
Node* Graph::sub(Node* a, Node *b) { IMPL_GRAPH_OP(Sub, a, b) }
Node* Graph::mul(Node* a, Node* b) { IMPL_GRAPH_OP(Mult, a, b) }
Node* Graph::div(Node* a, Node* b) { IMPL_GRAPH_OP(Div, a, b) }
Node* Graph::pow(Node* a, Node* b) { IMPL_GRAPH_OP(Pow, a, b) }
Node* Graph::max(Node* a, Node* b) { IMPL_GRAPH_OP(Max, a, b) }
Node* Graph::min(Node* a, Node* b) { IMPL_GRAPH_OP(Min, a, b) }
Node* Graph::log(Node* a) { IMPL_GRAPH_OP(Log, a) }
Node* Graph::minus(Node* a) { IMPL_GRAPH_OP(Minus, a) }
Node* Graph::abs(Node* a) { IMPL_GRAPH_OP(Abs, a) }
Node* Graph::relu(Node* a) { IMPL_GRAPH_OP(Relu, a) }
Node* Graph::sigmoid(Node* a) { IMPL_GRAPH_OP(Sigmoid, a) }
Node* Graph::tanh(Node* a) { IMPL_GRAPH_OP(Tanh, a) }
Node* Graph::sin(Node* a) { IMPL_GRAPH_OP(Sin, a) }
Node* Graph::cos(Node* a) { IMPL_GRAPH_OP(Cos, a) }

Graph::~Graph() {
    for (Node* node: nodes) { delete node; }
    for (OpNode* op: ops) { delete op; }
}

void Graph::clear_grad() { for (Node* node: nodes) { node->grad = 0; } }
void Graph::forward() { for (OpNode* op: ops) { op->forward(); } }
void Graph::backward(Node* node) {
    assert(node->graph == this);
    node->grad = 1;
    for (int i = ops.size() - 1; i >= 0; i--) {
        assert(ops[i]->output != nullptr);
        auto root_grad = ops[i]->output->grad;
        if (root_grad == 0) continue;
        ops[i]->backward(root_grad);
    }
}


std::string node_id(Node* node) { return std::to_string((size_t)node); }
std::string node_id(OpNode* op) { return std::to_string((size_t)op); }
std::string Graph::to_graphviz() {
    std::string t = "digraph G {\n";
    t += "  node [ shape=box, fixedsize=false, color=black, fontcolor=black, fontsize=12, fillcolor=white, style=filled ];\n";
    t += "  edge [ color=black ];\n";
    t += "  rankdir=TB;\n";
    t += "  nodesep=0.5;\n";

    auto drawOpNode = [&](OpNode* op) {
        t += "  " + node_id(op) + " [label=\"" + op->name + "\", color=blue];\n";
    };
    auto drawNode = [&](Node* node) {
        auto format_val = [](fp_t val) {
            std::stringstream ss;
            if (std::abs(val) < 1e-3) ss << std::scientific << std::setprecision(3) << val;
            else if (std::abs(val) > 1e3) ss << std::scientific << std::setprecision(3) << val;
            else ss << std::fixed << std::setprecision(2) << val;
            return ss.str();
        };
        auto get_node_label = [&format_val](Node* node) {
            std::string ret = "";
            if (node->name != "") ret += node->name + "@";
            ret = ret + format_val(node->value);
            if (node->requires_grad && node->grad != 0) ret += ", ∂=" + format_val(node->grad);
            if (!node->requires_grad) ret += ", const";
            return ret;
        };
        t += "  " + node_id(node) + " [label=\"" + get_node_label(node) + "\"];\n";
    };

    for (Node* node: nodes) {
        if (node->op != nullptr) continue;
        drawNode(node);
    }
    for (OpNode* op: ops) { 
        t += "subgraph cluster_" + node_id(op) + " {\n";
        t += "  margin=5;\n  bgcolor=lightgrey;\n";
        drawOpNode(op); 
        drawNode(op->output);
        t += "}\n";
    }

    for (OpNode* op: ops) {
        for (Node* input: op->inputs) { t += "  " + node_id(input) + " -> " + node_id(op) + ";\n"; }
        t += "  " + node_id(op) + " -> " + node_id(op->output) + "[color=blue];\n";
    }
    return t + "}";
}
}