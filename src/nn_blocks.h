/* Neural network building blocks */
#pragma once
#include "nn.h"
#include <random>
#include <cmath>

namespace nn{

enum class ActivationType {
    Relu,
    Sigmoid,
    Tanh,
};

template <size_t N>
struct ActivationLayer {
    std::shared_ptr<Node> input[N];
    std::shared_ptr<Node> output[N];
};

template <size_t N>
ActivationLayer<N> activation_layer(
    Graph& graph, 
    std::shared_ptr<Node> input[N],
    ActivationType type
){
    auto layer = ActivationLayer<N>();
    for (size_t i = 0; i < N; i++) {
        switch (type) {
            case ActivationType::Relu: layer.output[i] = graph.relu(input[i]); break;
            case ActivationType::Sigmoid: layer.output[i] = graph.sigmoid(input[i]); break;
            case ActivationType::Tanh: layer.output[i] = graph.tanh(input[i]); break;
            default: assert(false);
        }
    }
    return layer;
}

template <size_t N_in, size_t N_out>
struct LinearLayer {
    Graph* graph;
    std::shared_ptr<Node> input[N_in];
    std::shared_ptr<Node> output[N_out];
    std::shared_ptr<Node> weight[N_out][N_in];
    std::shared_ptr<Node> bias[N_out] = {nullptr};

    LinearLayer(Graph& graph): graph(&graph) {};

    LinearLayer with_bias() {
        for (size_t i = 0; i < N_out; i++) {
            bias[i] = graph->create_var(0, "bias_" + std::to_string(i));
            auto biased_out = graph->add(output[i], bias[i]);
            biased_out->name = output[i]->name + "_biased";
            output[i] = biased_out;
        }
        return *this;
    }

    LinearLayer normal_init(fp_t mean = true, fp_t sigma = 1) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<fp_t> dist(mean, sigma);

        for (auto &w : weight) {
            for (auto &v : w) {
                v->value = dist(gen);
            }
        }
        if (bias[0] != nullptr) {
            for (auto &b : bias) {
                b->value = dist(gen);
            }
        }
        return *this;
    }

    ActivationLayer<N_out> operator << (ActivationType t) {
        return activation_layer<N_out>(*graph, output, t);
    }
};

template <size_t N_in, size_t N_out>
LinearLayer<N_in, N_out> linear_layer(
    Graph& graph, 
    std::shared_ptr<Node> input[N_in],
    std::string name = ""
){
    static_assert(N_in > 0 && N_out > 0, "Invalid layer size");
    auto layer = LinearLayer<N_in, N_out>(graph);
    if (name == "") name = "linear_anon";

    for (size_t i = 0; i < N_in; i++) {
        layer.input[i] = input[i];
    }
    for (size_t i = 0; i < N_out; i++) {
        for (size_t j = 0; j < N_in; j++) {
            layer.weight[i][j] = graph.create_var(0, name + "_weight_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }
    for (size_t i = 0; i < N_out; i++) {
        layer.output[i] = graph.mul(layer.weight[i][0], input[0]);
        for (size_t j = 1; j < N_in; j++) {
            layer.output[i] = graph.add(layer.output[i], graph.mul(layer.weight[i][j], input[j]));
        }
        layer.output[i]->name = name + "_output_" + std::to_string(i);
    }
    return layer;
}

}
