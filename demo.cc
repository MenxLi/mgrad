#include "src/nn.h"
#include <random>
#include <iostream>
#include <fstream>
#include <ctime>

// The aimed function: f(x,y) = x^2 + 2y^2 + 1.25xy - 1
const float perterb = 0.1;
std::default_random_engine generator(time(0));
std::normal_distribution<float> sample_distribution(0, 2);
std::normal_distribution<float> eps_distribution(0, perterb);
float aim(float x, float y){
    float z = x*x + 2*y*y + 1.25*x*y - 1;
    z += eps_distribution(generator);
    return z;
};

struct Model{
    nn::Graph* graph;
    nn::Node* input_x;
    nn::Node* input_y;
    nn::Node* aim;
    nn::Node* prediciton;
    nn::Node* loss;
    std::vector<nn::Node*> params;
};

// a simple quadratic function fit
Model create_model(nn::Graph& graph){
    // names are optional, but useful for visualization
    nn::Node& input_x = graph.create_const(0, "x");
    nn::Node& input_y = graph.create_const(0, "y");
    nn::Node& aim = graph.create_const(0, "aim");

    nn::Node& a = graph.create_var(0, "a");
    nn::Node& b = graph.create_var(0, "b");
    nn::Node& c = graph.create_var(0, "c");
    nn::Node& d = graph.create_var(0, "d");
    nn::Node& e = graph.create_var(0, "e");
    nn::Node& f = graph.create_var(0, "f");
    nn::Node& pred = a * input_x.pow(2) + b * input_y.pow(2) + c * input_x * input_y + d * input_x + e * input_y + f;
    nn::Node& loss = (pred - aim).abs();
    pred.name = "pred"; 
    loss.name = "loss";
    return Model{&graph, &input_x, &input_y, &aim, &pred, &loss, {&a, &b, &c, &d, &e, &f}};
}

void train_step(Model& model, int n_iter, int total_iter){
    // one cycle lr
    const float base_lr = 3e-4;
    float lr = base_lr * (
        1 - std::abs(n_iter * 2.0 / total_iter - 1)
    );

    float x = sample_distribution(generator);
    float y = sample_distribution(generator);
    float z = aim(x, y);
    model.input_x->value = x;
    model.input_y->value = y;
    model.aim->value = z;
    model.graph->forward();
    model.graph->backward(model.loss);

    for (nn::Node* node: model.graph->nodes){
        if (node->requires_grad){
            node->value -= lr * node->grad;
        }
    }

    if (n_iter % 5000 == 0) {
        std::cout << "iter: " << n_iter << ", loss: " << model.loss->value << std::endl;
    }
    model.graph->clear_grad();
}

int main(){
    nn::Graph graph;

    Model model = create_model(graph);
    const int total_iter = 50000;
    for (int i = 0; i < total_iter; i++){
        train_step(model, i, total_iter);
    }

    std::cout << "final loss: " << model.loss->value << std::endl;
    std::cout << "final params for z = ax^2 + by^2 + cxy + dx + ey + f: \n\t";
    for (nn::Node* node: model.params){
        std::cout << ([](float x)->float{
            return int(x * 1000) / 1000.0;
        })(node->value) << " ";
    }
    std::cout << std::endl;

    // input a simple value and save the graph
    model.input_x->value = 1;
    model.input_y->value = 2;
    model.aim->value = aim(1, 2);
    model.graph->forward();

    std::ofstream ofs("model.mermaid");
    ofs << graph.to_mermaid();
    ofs.close();
    
    return 0;
}