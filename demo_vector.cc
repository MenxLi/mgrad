#include "src/nn_vector.h"
#include <random>
#include <iostream>
#include <fstream>
#include <ctime>

// The aimed function: f(x,y) = sign(x^2 + 2y^2 + 1.25xy - 1)
std::default_random_engine generator(time(0));
std::normal_distribution<float> sample_distribution(0, 2);
float aim(float x, float y){
    float z = x*x + 2*y*y + 1.25*x*y - 1;
    return z > 0 ? 1 : 0;
};

struct Model{
    nn::Graph* graph;
    nn::Node* input_x;
    nn::Node* input_y;
    nn::Node* aim;
    nn::Node* prediciton;
    nn::Node* loss;
};

// build a simple neural network
Model create_model(nn::Graph& graph){
    std::vector<nn::Node*> params;
    auto& input_x = graph.create_const(0, "x");
    auto& input_y = graph.create_const(0, "y");
    auto& output_aim = graph.create_const(0, "aim");

    nn::Node* input[2] = {
        &input_x,
        &input_y
    };

    auto l1 = nn::create_linear_layer<2, 4>(graph, input, "l1");
    auto a1 = nn::create_activation_layer<4>(graph, l1.output, nn::ActivationType::Relu);
    auto l2 = nn::create_linear_layer<4, 1>(graph, a1.output, "l2");
    auto a2 = nn::create_activation_layer<1>(graph, l2.output, nn::ActivationType::Sigmoid);

    auto& prediciton = *a2.output[0];
    auto& loss = (prediciton - output_aim).pow(2);
    return Model{
        &graph,
        &input_x,
        &input_y,
        &output_aim,
        &prediciton,
        &loss
    };
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

    if (n_iter % (int)1e4 == 0) {
        auto get_acc = [&]()->float{
            const int n_sample = 1000;
            int n_correct = 0;
            for (int i = 0; i < n_sample; i++){
                float x = sample_distribution(generator);
                float y = sample_distribution(generator);
                float z = aim(x, y);
                model.input_x->value = x;
                model.input_y->value = y;
                model.aim->value = z;
                model.graph->forward();
                if (model.prediciton->value > 0.5 == z > 0.5){
                    n_correct++;
                }
            }
            model.graph->clear_grad();
            return n_correct * 1.0 / n_sample;
        };

        std::cout << "iter: " << n_iter << ", loss: " << model.loss->value 
        << " | acc: " << get_acc() << std::endl;
    }
    model.graph->clear_grad();
}

int main(){
    nn::Graph graph;

    Model model = create_model(graph);
    const int total_iter = 1e5;
    for (int i = 0; i < total_iter; i++){
        train_step(model, i, total_iter);
    }

    std::cout << "final loss: " << model.loss->value << std::endl;

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