#include "nn.h"
#include <iostream>
#include <cmath>
#include <functional>

using namespace nn;

void m_assert(float a, float b) {
    if (abs(a - b) > 1e-4) {
        std::cout << "[Error] assertion failed: " << a << " != " << b << std::endl;
        exit(1);
    }
}

void test_unary_op(Graph& g, std::function<Node*(Node*)> fn, fp_t input, fp_t expected_grad) {
    Node& a = g.create_var(input);
    Node* b = fn(&a);
    g.forward();
    g.backward(b);
    m_assert(a.grad, expected_grad);
    g.clear_grad();
}

void test_binary_op(Graph& g, std::function<Node*(Node*, Node*)> fn, fp_t input1, fp_t input2, fp_t expected_grad1, fp_t expected_grad2) {
    Node& a = g.create_var(input1);
    Node& b = g.create_var(input2);
    Node* c = fn(&a, &b);
    g.forward();
    g.backward(c);
    m_assert(a.grad, expected_grad1);
    m_assert(b.grad, expected_grad2);
    g.clear_grad();
}


void t0() {
    Graph g;
    test_unary_op(
        g, [&g](Node* n) { return g.minus(n); }, 
        2, -1
        );
    test_unary_op(
        g, [&g](Node* n) { return g.inv(n); },
        2, -0.25
        );
    test_unary_op(
        g, [&g](Node* n) { return g.relu(n); },
        2, 1
        );
    test_unary_op(
        g, [&g](Node* n) { return g.relu(n); },
        -2, 0
        );
    test_unary_op(
        g, [&g](Node* n) { return g.sigmoid(n); },
        2, 0.1049935854035065
        );
    test_unary_op(
        g, [&g](Node* n) { return g.abs(n) ; },
        2, 1
    );
    test_unary_op(
        g, [&g](Node* n) { return g.abs(n) ; },
        -2, -1
    );

    test_binary_op(
        g, [&g](Node* a, Node* b) { return g.add(a, b); },
        2, 3, 1, 1
    );
    test_binary_op(
        g, [&g](Node* a, Node* b) { return g.sub(a, b); },
        2, 3, 1, -1
    );
    test_binary_op(
        g, [&g](Node* a, Node* b) { return g.mul(a, b); },
        2, 3, 3, 2
    );
    test_binary_op(
        g, [&g](Node* a, Node* b) { return g.div(a, b); },
        2, 3, 1./3, -2./9
    );
    test_binary_op(
        g, [&g](Node* a, Node* b) { return g.pow(a, b); },
        2, 3, 3*pow(2, 2), pow(2, 3)*log(2)
    );

}

int main(){
    t0();
    std::cout << "Test Passed." << std::endl;
    return 0;
}
