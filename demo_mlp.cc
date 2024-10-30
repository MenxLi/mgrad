#include "src/nn_blocks.h"
#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>
#include <ctime>
#include <tuple>

// a square function on zero level set
float aim_levelset(float x, float y){
    return std::cos(x) + std::sin(y);
};

std::default_random_engine generator(time(0));
template <size_t N>
std::array<std::tuple<float, float, float>, N> get_sample(){
    std::array<std::tuple<float, float, float>, N> samples;
    std::uniform_real_distribution<float> sample_distribution(-5, 5);
    for (size_t i = 0; i < N; i++){
        auto x = sample_distribution(generator);
        auto y = sample_distribution(generator);
        // take zero level set as classification
        auto z_raw = aim_levelset(x, y);
        auto z = static_cast<float>(z_raw < 0);
        samples[i] = std::make_tuple(x, y, z);
    }
    return samples;
}

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

    const int w = 4;
    // relu must use random initialization
    auto l1 = nn::linear_layer<2, w*2>(graph, input, "l1")
        .radom_init().with_bias() << nn::ActivationType::Relu;
    auto l2 = nn::linear_layer<w*2, w>(graph, l1.output, "l2")
        .radom_init().with_bias() << nn::ActivationType::Relu;
    auto l3 = nn::linear_layer<w, 1>(graph, l2.output, "l3")
        .radom_init().with_bias() << nn::ActivationType::Sigmoid;

    auto& prediciton = *l3.output[0];
    prediciton.name = "prediction";
    auto& bce_loss = -output_aim * prediciton.log() - (1 - output_aim) * (1 - prediciton).log();

    return Model{
        &graph,
        &input_x,
        &input_y,
        &output_aim,
        &prediciton,
        &bce_loss
    };
}

auto get_acc = [](Model& model)->float{
    int n_correct = 0;
    auto samples = get_sample<300>();
    for (auto [x, y, z] : samples){
        model.input_x->value = x;
        model.input_y->value = y;
        model.aim->value = z;
        model.graph->forward();
        if (model.prediciton->value > 0.5 == z > 0.5){
            n_correct++;
        }
    }
    model.graph->clear_grad();
    return n_correct * 1.0 / 300;
};

void train_step(Model& model, int n_iter, int total_iter){
    const float lr = 1e-2;
    const int batch_size = 64;

    nn::fp_t loss = 0;
    std::vector<nn::fp_t> grad_sums = std::vector<nn::fp_t>(model.graph->nodes.size(), 0);
    auto samples = get_sample<batch_size>();
    for (auto [x, y, z] : samples){
        model.input_x->value = x;
        model.input_y->value = y;
        model.aim->value = z;
        model.graph->forward();
        model.graph->backward(model.loss);
        for (std::size_t i = 0; i < model.graph->nodes.size(); i++){
            grad_sums[i] += model.graph->nodes[i]->grad;
        }
        loss += model.loss->value;
        model.graph->clear_grad();
    }

    for (std::size_t i = 0; i < model.graph->nodes.size(); i++){
        if (!model.graph->nodes[i]->requires_grad) continue;
        const nn::fp_t clip_threshold = 1e3;
        auto grad = grad_sums[i] / batch_size;
        if (grad > clip_threshold) grad = clip_threshold;
        if (grad < -clip_threshold) grad = -clip_threshold;
        model.graph->nodes[i]->value -= lr * grad;
    }

    if ((n_iter + 1) % (int)1e3 == 0) {
        std::cout << "Iteration [" << n_iter + 1 << "/" << total_iter << "]"
        << "\t loss: " << loss / batch_size << std::endl;
    }
    model.graph->clear_grad();
}

int main(){
    nn::Graph graph;

    Model model = create_model(graph);
    const int total_iter = 5e4;
    for (int i = 0; i < total_iter; i++){
        train_step(model, i, total_iter);
    }

    std::cout << "final loss: " << model.loss->value << ", acc: " << get_acc(model) << std::endl;

    // input a simple value and save the graph
    model.input_x->value = 1;
    model.input_y->value = 2;
    model.aim->value = aim_levelset(1, 2);
    model.graph->forward();

    std::ofstream ofs("model.mermaid");
    ofs << graph.to_mermaid();
    ofs.close();
    
    return 0;
}