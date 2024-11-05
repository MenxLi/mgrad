#include "src/nn.h"
#include <fstream>
#include <iostream>

#define ASSERT(x) if (!(x)) { std::cerr << "Assertion failed: " << #x << std::endl; exit(1); }

int main(){
    nn::Graph graph;

    // name is optional, for visualization only
    auto a = graph.variable(1, "a");
    auto b = graph.variable(2, "b");

    auto c = 2 * a + b;
    auto d = a * b;

    auto e = c + (d - 1);

    // e = (2a + b) + (a * b - 1) -> ∂e/∂a = 2 + b, ∂e/∂b = 1 + a
    graph.forward();
    graph.backward(e);

    ASSERT(a.grad() == 2 + b.value());
    ASSERT(b.grad() == 1 + a.value());

    // save the computation graph to graphviz format
    std::ofstream file("model.gv");
    file << graph.to_graphviz();
    file.close();

    std::cout << "Success, check computational graph: 'model.gv'" << std::endl;
    
    return 0;
}