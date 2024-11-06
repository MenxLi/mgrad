#pragma once
#include <string>
#include <vector>
#include <cassert>
#include <functional>

namespace nn {

typedef double fp_t;

struct Node;
struct Graph;
struct NodeProxy;

struct OpNodeBase {
    std::string name = "Op";
    std::vector<Node*> inputs = std::vector<Node*>();
    Node *output = nullptr;
    virtual ~OpNodeBase() {}
};

// the derived class should implement:
//  - void forward()
//  - void backward(fp_t grad)
// forward_call update output->value
// backward_call update inputs[.]->grad
template <typename Derived>
struct OpNode: public OpNodeBase {
    inline void forward_call() { static_cast<Derived*>(this)->forward(); }
    inline void backward_call(fp_t grad) { static_cast<Derived*>(this)->backward(grad); }
    virtual ~OpNode() {}
};

#define DECLARE_OP(OP) \
    struct Op##OP: public OpNode<Op##OP> { \
        Op##OP() { name = #OP; } \
        void forward(); \
        void backward(fp_t grad); \
    };

DECLARE_OP(Add)
DECLARE_OP(Sub)
DECLARE_OP(Mult)
DECLARE_OP(Div)
DECLARE_OP(Pow)
DECLARE_OP(Max)
DECLARE_OP(Min)
DECLARE_OP(Log)
DECLARE_OP(Minus)    // unary minus
DECLARE_OP(Abs)
DECLARE_OP(Sin)
DECLARE_OP(Cos)
DECLARE_OP(Relu)
DECLARE_OP(Sigmoid)
DECLARE_OP(Tanh)

struct Graph {

    struct OpRecord {
        OpNodeBase* op;
        Node* output;
        std::function<void()> forward;
        std::function<void(fp_t)> backward;
    };

    ~Graph();
    std::vector<Node*> nodes;
    std::vector<OpRecord> op_records;
    void forward();
    void backward(Node* node);
    void backward(NodeProxy node_proxy);
    void clear_grad();

    NodeProxy variable(fp_t value = 0, std::string name = "");
    NodeProxy constant(fp_t value = 0, std::string name = "");

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
    Node* sin(Node* a);
    Node* cos(Node* a);
    Node* abs(Node* a);

    Node* relu(Node* a);
    Node* sigmoid(Node* a);
    Node* tanh(Node* a);

    std::string to_graphviz();

private:
    Graph& operator=(const Graph& b) = delete;
    Graph& operator=(const Graph&& b) = delete;
};

struct Node {
    fp_t value;
    fp_t grad = 0;
    Graph* graph = nullptr;
    std::string name = "";
    bool is_leaf = true;
    bool requires_grad = true;
    Node(Graph* g, std::string name = "", fp_t value = 0): value(value), graph(g), name(name) {}

private:
    Node(const Node& b) = delete;
    Node(const Node&& b) = delete;
    Node& operator=(const Node& b) = delete;
    Node& operator=(const Node&& b) = delete;
};


// For handy usage of: 
// 1. operator overloading
// 2. copy and move assignment
struct NodeProxy{
    Node* ptr;

    NodeProxy(Node* ptr): ptr(ptr) {}
    NodeProxy(Node& node): ptr(&node) {}
    NodeProxy(const NodeProxy& b): ptr(b.ptr) {}
    NodeProxy(const NodeProxy&& b): ptr(b.ptr) {}
    NodeProxy& operator=(const NodeProxy& b) { ptr = b.ptr; return *this; }

    inline Graph& graph() { return *ptr->graph; }
    inline bool requires_grad() { return ptr->requires_grad; }
    inline void set_value(fp_t v) { ptr->value = v; }
    inline fp_t value() { return ptr->value; }
    inline fp_t grad() { return ptr->grad; }
 
    NodeProxy operator-() { return NodeProxy(graph().minus(ptr)); }
    NodeProxy operator+(NodeProxy b) { return NodeProxy(graph().add(ptr, b.ptr)); }
    NodeProxy operator-(NodeProxy b) { return NodeProxy(graph().sub(ptr, b.ptr)); }
    NodeProxy operator*(NodeProxy b) { return NodeProxy(graph().mul(ptr, b.ptr)); }
    NodeProxy operator/(NodeProxy b) { return NodeProxy(graph().div(ptr, b.ptr)); }
    NodeProxy operator+(fp_t b) { return NodeProxy(graph().add(ptr, graph().create_const(b))); }
    NodeProxy operator-(fp_t b) { return NodeProxy(graph().sub(ptr, graph().create_const(b))); }
    NodeProxy operator*(fp_t b) { return NodeProxy(graph().mul(ptr, graph().create_const(b))); }
    NodeProxy operator/(fp_t b) { return NodeProxy(graph().div(ptr, graph().create_const(b))); }

    NodeProxy pow(NodeProxy b) { return NodeProxy(graph().pow(ptr, b.ptr)); }
    NodeProxy pow(fp_t b) { return NodeProxy(graph().pow(ptr, graph().create_const(b))); }
    NodeProxy max(NodeProxy b) { return NodeProxy(graph().max(ptr, b.ptr)); }
    NodeProxy max(fp_t b) { return NodeProxy(graph().max(ptr, graph().create_const(b))); }
    NodeProxy min(NodeProxy b) { return NodeProxy(graph().min(ptr, b.ptr)); }
    NodeProxy min(fp_t b) { return NodeProxy(graph().min(ptr, graph().create_const(b))); }
    NodeProxy log() { return NodeProxy(graph().log(ptr)); }
    NodeProxy abs() { return NodeProxy(graph().abs(ptr)); }
    NodeProxy sin() { return NodeProxy(graph().sin(ptr)); }
    NodeProxy cos() { return NodeProxy(graph().cos(ptr)); }

    NodeProxy relu() { return NodeProxy(graph().relu(ptr)); }
    NodeProxy sigmoid() { return NodeProxy(graph().sigmoid(ptr)); }
    NodeProxy tanh() { return NodeProxy(graph().tanh(ptr)); }
};
inline NodeProxy operator+(fp_t a, NodeProxy b) { return NodeProxy(*b.graph().add(b.graph().create_const(a), b.ptr)); }
inline NodeProxy operator-(fp_t a, NodeProxy b) { return NodeProxy(*b.graph().sub(b.graph().create_const(a), b.ptr)); }
inline NodeProxy operator*(fp_t a, NodeProxy b) { return NodeProxy(*b.graph().mul(b.graph().create_const(a), b.ptr)); }
inline NodeProxy operator/(fp_t a, NodeProxy b) { return NodeProxy(*b.graph().div(b.graph().create_const(a), b.ptr)); }

}