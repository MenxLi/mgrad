#pragma once
#include <string>
#include <vector>
#include <cassert>

namespace nn {

typedef double fp_t;

struct Node;
struct Graph;

struct OpNode {
    std::string name = "Op";
    std::vector<Node*> inputs = std::vector<Node*>();
    Node *output = nullptr;
    virtual void forward() = 0;             // update output->value
    virtual void backward() = 0;            // update inputs[.]->grad, using output->grad
    fp_t root_grad();
    virtual ~OpNode() {}
};

#define DECLEAR_OP(OP) \
    struct Op##OP: public OpNode { \
        Op##OP() { name = #OP; } \
        void forward() override; \
        void backward() override; \
    };

DECLEAR_OP(Add)
DECLEAR_OP(Sub)
DECLEAR_OP(Mult)
DECLEAR_OP(Div)
DECLEAR_OP(Pow)
DECLEAR_OP(Minus)    // unary minus
DECLEAR_OP(Inv)      // unary inverse
DECLEAR_OP(Abs)
DECLEAR_OP(Relu)
DECLEAR_OP(Sigmoid)

struct Graph {

    ~Graph();
    std::vector<Node*> nodes;
    std::vector<OpNode*> ops;
    void forward();
    void backward(Node* node);
    void clear_grad();

    Node& create_var(fp_t value, std::string name = "");
    Node& create_const(fp_t value, std::string name = "");
    Node* add(Node* a, Node* b);
    Node* sub(Node* a, Node* b);
    Node* mul(Node* a, Node* b);
    Node* div(Node* a, Node* b);
    Node* pow(Node* a, Node* b);

    Node* minus(Node* a);
    Node* inv(Node* a);
    Node* abs(Node* a);

    Node* relu(Node* a);
    Node* sigmoid(Node* a);

    std::string to_mermaid();
};

struct Node {
    fp_t value;
    fp_t grad = 0;
    Graph* graph = nullptr;
    OpNode* op = nullptr;
    std::string name = "";
    bool requires_grad = true;
    Node(Graph* g, std::string name = "", fp_t value = 0): value(value), graph(g), name(name) {}

    Node& operator-() { return *graph->minus(this); }
    Node& operator+(Node& b) { return *graph->add(this, &b); }
    Node& operator-(Node& b) { return *graph->sub(this, &b); }
    Node& operator*(Node& b) { return *graph->mul(this, &b); }
    Node& operator/(Node& b) { return *graph->div(this, &b); }
    Node& operator+(fp_t b) { return *graph->add(this, &graph->create_const(b)); }
    Node& operator-(fp_t b) { return *graph->sub(this, &graph->create_const(b)); }
    Node& operator*(fp_t b) { return *graph->mul(this, &graph->create_const(b)); }
    Node& operator/(fp_t b) { return *graph->div(this, &graph->create_const(b)); }

    Node& pow(Node& b) { return *graph->pow(this, &b); }
    Node& pow(fp_t b) { return *graph->pow(this, &graph->create_const(b)); }
    Node& abs() { return *graph->abs(this); }
    Node& relu() { return *graph->relu(this); }
    Node& sigmoid() { return *graph->sigmoid(this); }

private:
    Node(const Node& b) = delete;
    Node(const Node&& b) = delete;
    Node& operator=(const Node& b) = delete;
    Node& operator=(const Node&& b) = delete;
};

inline Node& operator+(fp_t a, Node& b) { return *b.graph->add(&b.graph->create_const(a), &b); }
inline Node& operator-(fp_t a, Node& b) { return *b.graph->sub(&b.graph->create_const(a), &b); }
inline Node& operator*(fp_t a, Node& b) { return *b.graph->mul(&b.graph->create_const(a), &b); }
inline Node& operator/(fp_t a, Node& b) { return *b.graph->div(&b.graph->create_const(a), &b); }


/* -------- Template Implementations for Neural Network -------- */
template <size_t N_in, size_t N_out>
struct LinearLayer {
    Node* input[N_in];
    Node* output[N_out];
    Node* weight[N_out][N_in];
    Node* bias[N_out];
};

template <size_t N_in, size_t N_out>
LinearLayer<N_in, N_out> linear_layer(
    Graph& graph, 
    Node* input[N_in],
    const std::string& name = ""
){
    auto layer = LinearLayer<N_in, N_out>();

    for (size_t i = 0; i < N_in; i++) {
        layer.input[i] = input[i];
    }
    for (size_t i = 0; i < N_out; i++) {
        layer.bias[i] = &graph.create_var(0, name + "_bias_" + std::to_string(i));
        for (size_t j = 0; j < N_in; j++) {
            layer.weight[i][j] = &graph.create_var(0, name + "_weight_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }
    for (size_t i = 0; i < N_out; i++) {
        layer.output[i] = layer.bias[i];
        for (size_t j = 0; j < N_in; j++) {
            auto& w = *layer.weight[i][j];
            auto& x = *input[j];
            layer.output[i] = &(*layer.output[i] + w * x);
        }
    }
    return layer;
}

enum class ActivationType {
    Relu,
    Sigmoid,
};

template <size_t N>
struct ActivationLayer {
    Node* input[N];
    Node* output[N];
};

template <size_t N>
ActivationLayer<N> activation_layer(
    Graph& graph, 
    Node* input[N],
    ActivationType type
){
    auto layer = ActivationLayer<N>();
    for (size_t i = 0; i < N; i++) {
        switch (type) {
            case ActivationType::Relu:
                layer.output[i] = graph.relu(input[i]);
                break;
            case ActivationType::Sigmoid:
                layer.output[i] = graph.sigmoid(input[i]);
                break;
            default:
                assert(false);
        }
    }
    return layer;
}

}