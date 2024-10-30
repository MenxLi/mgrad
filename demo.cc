#include "src/nn.h"
#include <fstream>
#include <iostream>

#define ASSERT(x) if (!(x)) { std::cerr << "Assertion failed: " << #x << std::endl; exit(1); }

int main(){
    nn::Graph graph;

    // name is optional, for visualization only
    // be sure to use & to reference the variables
    auto& a = graph.create_var(1, "a");
    auto& b = graph.create_var(2, "b");

    auto& c = 2 * a + b;
    auto& d = a * b;

    auto& e = c + (d - 1);

    // e = (2a + b) + (a * b - 1)
    // -> ∂e/∂a = 2 + b, ∂e/∂b = 1 + a
    graph.forward();
    graph.backward(&e);

    ASSERT(a.grad == 2 + b.value);
    ASSERT(b.grad == 1 + a.value);

    // save the computation graph to mermaid format
    std::ofstream file("model.mermaid");
    file << graph.to_mermaid();
    file.close();

    std::cout << "Success, check model.mermaid" << std::endl;
    
    return 0;
}