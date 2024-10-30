#include "src/nn_blocks.h"
#include <random>
#include <iostream>
#include <fstream>
#include <ctime>

// The aimed function: f(x,y) = sign(x^2 + 2y^2 + 1.25xy - 1)
std::default_random_engine generator(time(0));
std::normal_distribution<float> sample_distribution(0, 5);
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

    // relu must use random initialization
    auto l1 = nn::linear_layer<2, 8>(graph, input, "l1")
        .radom_init().with_bias() << nn::ActivationType::Relu;
    auto l2 = nn::linear_layer<8, 4>(graph, l1.output, "l2")
        .radom_init().with_bias() << nn::ActivationType::Sigmoid;
    auto l3 = nn::linear_layer<4, 4>(graph, l2.output, "l3")
        .radom_init().with_bias() << nn::ActivationType::Relu;
    auto l4 = nn::linear_layer<4, 1>(graph, l3.output, "l4") << nn::ActivationType::Sigmoid;

    auto& prediciton = *l4.output[0];
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
    const int batch_size = 16;

    // keep a record of gradients, for the sake of batch gradient descent
    std::vector<nn::fp_t> grads = std::vector<nn::fp_t>(model.graph->nodes.size(), 0);
    nn::fp_t loss = 0;
    for (int i = 0; i < batch_size; i++){
        float x = sample_distribution(generator);
        float y = sample_distribution(generator);
        float z = aim(x, y);
        model.input_x->value = x;
        model.input_y->value = y;
        model.aim->value = z;
        model.graph->forward();
        model.graph->backward(model.loss);
        for (std::size_t i = 0; i < model.graph->nodes.size(); i++){
            grads[i] += model.graph->nodes[i]->grad;
        }
        loss += model.loss->value;
        model.graph->clear_grad();
    }

    for (std::size_t i = 0; i < model.graph->nodes.size(); i++){
        if (!model.graph->nodes[i]->requires_grad) continue;
        // batch averaging and gradient clipping
        auto grad = grads[i] / batch_size;
        const float threshold = 1;
        if (grad > threshold) grad = threshold;
        if (grad < -threshold) grad = -threshold;
        model.graph->nodes[i]->value -= lr * grad;
    }

    if (n_iter % (int)1e3 == 0) {
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

        std::cout << "iter: " << n_iter << ", loss: " << loss / batch_size 
        << ", acc: " << get_acc() << std::endl;
    }
    model.graph->clear_grad();
}

int main(){
    nn::Graph graph;

    Model model = create_model(graph);
    const int total_iter = 1e4;
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