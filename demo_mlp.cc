#include "src/nn.h"
#include "src/nn_blocks.h"
#include "utils/bitmap.h"

#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>
#include <array>

using nn::fp_t;

float aim_levelset(float x, float y){
    auto rotate = [](float x, float y, float theta){
        auto rx = std::cos(theta) * x - std::sin(theta) * y;
        auto ry = std::sin(theta) * x + std::cos(theta) * y;
        return std::make_pair(rx, ry);
    };
    auto oval = [](float x, float y, float a, float b, float scale = 1){
        return x * x / (a * a) + y * y / (b * b) - scale;
    };
    auto [x1, y1] = rotate(x, y, 0.2);
    auto [x2, y2] = rotate(x, y, -0.6);
    return std::min(
        oval(x1 + 1.5, y1 - 1, 1, 2, 1.2), 
        oval(x2 - 0.5, y2 + 2, 2, 1, 0.8)
        );
};

std::random_device rd;
std::mt19937 generator(rd());
template <size_t N>
std::array<fp_t[3], N> get_samples(){
    std::array<fp_t[3], N> samples;
    std::uniform_real_distribution<fp_t> sample_distribution(-5, 5);
    for (size_t i = 0; i < N; i++){
        // take zero level set as classification
        auto x = sample_distribution(generator);
        auto y = sample_distribution(generator);
        auto z_raw = aim_levelset(x, y);
        auto z = static_cast<float>(z_raw < 0);
        samples[i][0] = x; samples[i][1] = y; samples[i][2] = z;
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

    const int w = 8;
    auto l1 = nn::linear_layer<2, 2*w>(graph, input, "l1")
        .with_bias().normal_init() << nn::ActivationType::Tanh;
    auto l2 = nn::linear_layer<2*w, w>(graph, l1.output, "l2")
        .with_bias().normal_init() << nn::ActivationType::Relu;
    auto l3 = nn::linear_layer<w, w>(graph, l2.output, "l3")
        .with_bias().normal_init() << nn::ActivationType::Relu;
    auto l4 = nn::linear_layer<w, 1>(graph, l3.output, "l4")
        .with_bias().normal_init() << nn::ActivationType::Sigmoid;

    auto& prediciton = *l4.output[0];
    prediciton.name = "prediction";
    const nn::fp_t eps = 1e-7;
    auto& bce_loss = -output_aim * (prediciton + eps).log() - (1 - output_aim) * (1 - prediciton + eps).log();

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
    float n_correct = 0;
    const int n_samples = 500;
    for (auto [x, y, z] : get_samples<n_samples>()){
        model.input_x->value = x;
        model.input_y->value = y;
        model.aim->value = z;
        model.graph->forward();
        if ((model.prediciton->value > 0.5) == (z > 0.5)){
            n_correct++;
        }
    }
    model.graph->clear_grad();
    return n_correct / static_cast<float>(n_samples);
};

void train_step(Model& model, int n_iter, int total_iter){
    float lr = 1e-2;
    const int batch_size = 32;

    fp_t loss = 0;
    std::vector<fp_t> grad_sums = std::vector<fp_t>(model.graph->nodes.size(), 0);
    for (auto [x, y, z] : get_samples<batch_size>()){
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
        const fp_t clip_threshold = 1e3;
        auto grad = grad_sums[i] / batch_size;
        if (grad > clip_threshold) grad = clip_threshold;
        if (grad < -clip_threshold) grad = -clip_threshold;
        model.graph->nodes[i]->value -= lr * grad;
    }

    if ((n_iter + 1) % (int)1e4 == 0) {
        std::cout << "Iteration [" << n_iter + 1 << "/" << total_iter << "]"
        << ", loss: " << loss / batch_size << std::endl;
    }
    model.graph->clear_grad();
}

void save_bitmap(Model& model);

int main(){
    nn::Graph graph;

    Model model = create_model(graph);
    const int total_iter = 8e4;
    for (int i = 0; i < total_iter; i++){
        train_step(model, i, total_iter);
    }
    std::cout << "final loss: " << model.loss->value << ", acc: " << get_acc(model) << std::endl;

    save_bitmap(model);
    return 0;
}

// ================== Utility for visualization ==================
void save_bitmap(Model& model){
    const int w = 256;
    const int h = 256;
    float res[w][h];

    auto norm_value = [](nn::fp_t v){
        // map to -1 - 2, for visualization
        if (v < -1) v = -1;
        if (v > 2) v = 2;
        v = (v + 1) / 3;
        return (unsigned char)(v * 255);
    };

    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            model.input_x->value = i * 10.0 / w - 5;
            model.input_y->value = j * 10.0 / h - 5;
            model.graph->forward();
            auto pred = model.prediciton->value;
            res[i][j] = norm_value(pred);
        }
    }
    write_bitmap<w, h>("mlp_prediction.bmp", res, res, res);

    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            auto z = static_cast<nn::fp_t>(
                aim_levelset(i * 10.0 / w - 5, j * 10.0 / h - 5) < 0
                );
            res[i][j] = norm_value(z);
            res[i][j] = norm_value(z);
            res[i][j] = norm_value(z);
        }
    }
    write_bitmap<w, h>("mlp_aim.bmp", res, res, res);
}
