#include "nn.h"
#include <cassert>
#include <iostream>

// cases from: https://github.com/kennysong/minigrad/blob/master/tests.ipynb
using namespace nn;

NodeProxy f1(NodeProxy a, NodeProxy b){
    auto c = a + b;
    auto d = a * b + b.pow(3);
    c = c + (c + 1);
    c = c + 1 + c + (-a);
    d = d + d * 2 + (b+a).relu();
    d = d + 3 * d + (b-a).relu();
    auto e = c - d;
    auto f = e.pow(2);
    auto g = f/2;
    auto g1 = g + 10/f;
    return g1;
}

NodeProxy f0(NodeProxy x, NodeProxy y){
    auto n4 = x * x;
    auto n5 = n4 * y;
    auto n6 = y + 2;
    auto n7 = n5 + n6;
    return n7;
}

void assert_close(fp_t a, fp_t b){
    if (a - b > 1e-4){
        std::cout << "[Error] assertion failed: " << a << " != " << b << std::endl;
        exit(1);
    }
}

int main(){

    auto test = [](
        NodeProxy (*f)(NodeProxy, NodeProxy),
        fp_t a, fp_t b, fp_t expected_a, fp_t expected_b
        ){
        Graph graph;
        auto na = graph.variable(a, "a");
        auto nb = graph.variable(b, "b");
        auto result = f(na, nb);
        graph.forward();
        graph.backward(result);
        assert_close(na.grad(), expected_a);
        assert_close(nb.grad(), expected_b);
    };

    test(f0, 3, 4, 24, 10);
    test(f1, -4, 2, 138.8338, 645.5773);
    std::cout << "Test passed." << std::endl;

}