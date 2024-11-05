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
    virtual void forward() = 0;                     // update output->value
    virtual void backward(fp_t grad) = 0;           // update inputs[.]->grad
    virtual ~OpNode() {}
};

#define DECLEAR_OP(OP) \
    struct Op##OP: public OpNode { \
        Op##OP() { name = #OP; } \
        void forward() override; \
        void backward(fp_t grad) override; \
    };

DECLEAR_OP(Add)
DECLEAR_OP(Sub)
DECLEAR_OP(Mult)
DECLEAR_OP(Div)
DECLEAR_OP(Pow)
DECLEAR_OP(Max)
DECLEAR_OP(Min)
DECLEAR_OP(Log)
DECLEAR_OP(Minus)    // unary minus
DECLEAR_OP(Abs)
DECLEAR_OP(Relu)
DECLEAR_OP(Sigmoid)
DECLEAR_OP(Tanh)
DECLEAR_OP(Sin)
DECLEAR_OP(Cos)

struct Graph {

    ~Graph();
    std::vector<Node*> nodes;
    std::vector<OpNode*> ops;
    void forward();
    void backward(Node* node);
    void clear_grad();

    Node* create_var(fp_t value = 0, std::string name = "");
    Node* create_const(fp_t value = 0, std::string name = "");
    Node* add(Node* a, Node* b);
    Node* sub(Node* a, Node* b);
    Node* mul(Node* a, Node* b);
    Node* div(Node* a, Node* b);
    Node* pow(Node* a, Node* b);
    Node* max(Node* a, Node* b);
    Node* min(Node* a, Node* b);

    Node* log(Node* a);
    Node* minus(Node* a);
    Node* abs(Node* a);

    Node* relu(Node* a);
    Node* sigmoid(Node* a);
    Node* tanh(Node* a);

    Node* sin(Node* a);
    Node* cos(Node* a);

    std::string to_graphviz();
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
    Node& operator+(fp_t b) { return *graph->add(this, graph->create_const(b)); }
    Node& operator-(fp_t b) { return *graph->sub(this, graph->create_const(b)); }
    Node& operator*(fp_t b) { return *graph->mul(this, graph->create_const(b)); }
    Node& operator/(fp_t b) { return *graph->div(this, graph->create_const(b)); }

    Node& pow(Node& b) { return *graph->pow(this, &b); }
    Node& pow(fp_t b) { return *graph->pow(this, graph->create_const(b)); }
    Node& max(Node& b) { return *graph->max(this, &b); }
    Node& max(fp_t b) { return *graph->max(this, graph->create_const(b)); }
    Node& min(Node& b) { return *graph->min(this, &b); }
    Node& min(fp_t b) { return *graph->min(this, graph->create_const(b)); }
    Node& log() { return *graph->log(this); }
    Node& abs() { return *graph->abs(this); }
    Node& relu() { return *graph->relu(this); }
    Node& sigmoid() { return *graph->sigmoid(this); }
    Node& tanh() { return *graph->tanh(this); }
    Node& sin() { return *graph->sin(this); }
    Node& cos() { return *graph->cos(this); }

private:
    Node(const Node& b) = delete;
    Node(const Node&& b) = delete;
    Node& operator=(const Node& b) = delete;
    Node& operator=(const Node&& b) = delete;
};

inline Node& operator+(fp_t a, Node& b) { return *b.graph->add(b.graph->create_const(a), &b); }
inline Node& operator-(fp_t a, Node& b) { return *b.graph->sub(b.graph->create_const(a), &b); }
inline Node& operator*(fp_t a, Node& b) { return *b.graph->mul(b.graph->create_const(a), &b); }
inline Node& operator/(fp_t a, Node& b) { return *b.graph->div(b.graph->create_const(a), &b); }

}