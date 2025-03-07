#include "nn.h"
#include <iostream>
#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <iomanip>

namespace nn {

NodeProxy Graph::variable(fp_t value, std::string name) {
    std::shared_ptr<Node> node = create_var(value, name);
    return NodeProxy(node);
}

NodeProxy Graph::constant(fp_t value, std::string name) {
    std::shared_ptr<Node> node = create_const(value, name);
    return NodeProxy(node);
}

// create a new leaf node
std::shared_ptr<Node> Graph::create_var(fp_t value, std::string name) {
    std::shared_ptr<Node> node = std::make_shared<Node>(this, name, value);
    nodes.push_back(node);
    return node;
}
std::shared_ptr<Node> Graph::create_const(fp_t value, std::string name) {
    std::shared_ptr<Node> node = create_var(value, name);
    node->requires_grad = false;
    return node;
}

#define IMPL_GRAPH_OP(OP, ...) \
    std::shared_ptr<Op##OP> op = std::make_shared<Op##OP>(); \
    std::shared_ptr<Node> node = std::make_shared<Node>(this); \
    node->op = op; \
    op->inputs = {__VA_ARGS__}; \
    op->output = node; \
    ops.push_back(op); \
    nodes.push_back(node); \
    return node; \

std::shared_ptr<Node> Graph::add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { IMPL_GRAPH_OP(Add, a, b) }
std::shared_ptr<Node> Graph::sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { IMPL_GRAPH_OP(Sub, a, b) }
std::shared_ptr<Node> Graph::mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { IMPL_GRAPH_OP(Mult, a, b) }
std::shared_ptr<Node> Graph::div(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { IMPL_GRAPH_OP(Div, a, b) }
std::shared_ptr<Node> Graph::pow(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { IMPL_GRAPH_OP(Pow, a, b) }
std::shared_ptr<Node> Graph::max(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { IMPL_GRAPH_OP(Max, a, b) }
std::shared_ptr<Node> Graph::min(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { IMPL_GRAPH_OP(Min, a, b) }
std::shared_ptr<Node> Graph::log(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Log, a) }
std::shared_ptr<Node> Graph::minus(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Minus, a) }
std::shared_ptr<Node> Graph::abs(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Abs, a) }
std::shared_ptr<Node> Graph::relu(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Relu, a) }
std::shared_ptr<Node> Graph::sigmoid(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Sigmoid, a) }
std::shared_ptr<Node> Graph::tanh(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Tanh, a) }
std::shared_ptr<Node> Graph::sin(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Sin, a) }
std::shared_ptr<Node> Graph::cos(std::shared_ptr<Node> a) { IMPL_GRAPH_OP(Cos, a) }

Graph::~Graph() {
    for (std::shared_ptr<Node> node: nodes) { node.reset(); }
    for (std::shared_ptr<OpNode> op: ops) { op.reset(); }
}

void Graph::clear_grad() { for (std::shared_ptr<Node> node: nodes) { node->grad = 0; } }
void Graph::forward() { for (std::shared_ptr<OpNode> op: ops) { op->forward(); } }
void Graph::backward(std::shared_ptr<Node> node, fp_t grad) {
    assert(node->graph == this);
    node->grad = grad;
    for (int i = ops.size() - 1; i >= 0; i--) {
        assert(ops[i]->output != nullptr);
        auto root_grad = ops[i]->output->grad;
        if (root_grad == 0) continue;
        ops[i]->backward(root_grad);
    }
}


std::string node_id(std::shared_ptr<Node> node) { return std::to_string((size_t)node.get()); }
std::string node_id(std::shared_ptr<OpNode> op) { return std::to_string((size_t)op.get()); }
std::string Graph::to_graphviz() {
    std::string t = "digraph G {\n";
    t += "  node [ shape=box, fixedsize=false, color=black, fontcolor=black, fontsize=12, fillcolor=white, style=filled ];\n";
    t += "  edge [ color=black ];\n";
    t += "  rankdir=TB;\n";
    t += "  nodesep=0.5;\n";

    auto drawOpNode = [&](std::shared_ptr<OpNode> op) {
        t += "  " + node_id(op) + " [label=\"" + op->name + "\", color=blue];\n";
    };
    auto drawNode = [&](std::shared_ptr<Node> node) {
        auto format_val = [](fp_t val) {
            std::stringstream ss;
            if (std::abs(val) < 1e-3) ss << std::scientific << std::setprecision(3) << val;
            else if (std::abs(val) > 1e3) ss << std::scientific << std::setprecision(3) << val;
            else ss << std::fixed << std::setprecision(2) << val;
            return ss.str();
        };
        auto get_node_label = [&format_val](std::shared_ptr<Node> node) {
            std::string ret = "";
            if (node->name != "") ret += node->name + "@";
            ret = ret + format_val(node->value);
            if (node->requires_grad && node->grad != 0) ret += ", ∂=" + format_val(node->grad);
            if (!node->requires_grad) ret += ", const";
            return ret;
        };
        t += "  " + node_id(node) + " [label=\"" + get_node_label(node) + "\"];\n";
    };

    for (auto node: nodes) {
        if (node->op != nullptr) continue;
        drawNode(node);
    }
    for (auto op: ops) { 
        t += "subgraph cluster_" + node_id(op) + " {\n";
        t += "  margin=5;\n  bgcolor=lightgrey;\n";
        drawOpNode(op); 
        drawNode(op->output);
        t += "}\n";
    }

    for (auto op: ops) {
        for (auto input: op->inputs) { t += "  " + node_id(input) + " -> " + node_id(op) + ";\n"; }
        t += "  " + node_id(op) + " -> " + node_id(op->output) + "[color=blue];\n";
    }
    return t + "}";
}
}