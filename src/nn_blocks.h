#pragma once
#include "nn.h"
#include <cmath>

namespace nn{

/* -------- Template Implementations for Neural Network -------- */

enum class ActivationType {
    Relu,
    Sigmoid,
    Tanh,
};

template <size_t N>
struct ActivationLayer {
    Node* input[N];
    Node* output[N];
};

template <size_t N>
ActivationLayer<N> activation_layer(
    Graph& graph, 
    Node* input[N],
    ActivationType type
){
    auto layer = ActivationLayer<N>();
    for (size_t i = 0; i < N; i++) {
        switch (type) {
            case ActivationType::Relu:
                layer.output[i] = graph.relu(input[i]);
                break;
            case ActivationType::Sigmoid:
                layer.output[i] = graph.sigmoid(input[i]);
                break;
            case ActivationType::Tanh:
                layer.output[i] = graph.tanh(input[i]);
                break;
            default:
                assert(false);
        }
    }
    return layer;
}

template <size_t N_in, size_t N_out>
struct LinearLayer {
    Graph* graph;
    Node* input[N_in];
    Node* output[N_out];
    Node* weight[N_out][N_in];
    Node* bias[N_out] = {nullptr};

    LinearLayer(Graph& graph): graph(&graph) {};

    LinearLayer with_bias() {
        for (size_t i = 0; i < N_out; i++) {
            bias[i] = &graph->create_var(0, "bias_" + std::to_string(i));
            auto& biased_out = *output[i] + *bias[i];
            biased_out.name = output[i]->name + "_biased";
            output[i] = &biased_out;
        }
        return *this;
    }

    // variance only applies to normal distribution
    LinearLayer random_init(fp_t scale = 1, bool normal_dist = true, fp_t variance = 1) {
        auto rnd = [&scale, &normal_dist, &variance](){ 
            const fp_t PI = 3.14159265358979323846;
            auto uniform_one = static_cast<fp_t>((rand() % 10000) / 10000.0); 
            auto v = scale * (uniform_one * 2 - 1);
            if (normal_dist) {
                v = ( 1 / std::sqrt(2 * PI * std::pow(variance, 2)) ) 
                    * std::exp(-std::pow(v, 2) / (2 * std::pow(variance, 2)));
            }
            return v;
            };

        for (size_t i = 0; i < N_out; i++) {
            for (size_t j = 0; j < N_in; j++) {
                weight[i][j]->value = rnd();
            }
        }
        if (bias[0] != nullptr) {
            for (size_t i = 0; i < N_out; i++) {
                bias[i]->value = rnd();
            }
        }
        return *this;
    }

    ActivationLayer<N_out> operator << (ActivationType type) {
        return activation_layer<N_out>(*graph, output, type);
    }
};

template <size_t N_in, size_t N_out>
LinearLayer<N_in, N_out> linear_layer(
    Graph& graph, 
    Node* input[N_in],
    const std::string& name = ""
){
    static_assert(N_in > 0 && N_out > 0, "Invalid layer size");
    auto layer = LinearLayer<N_in, N_out>(graph);

    for (size_t i = 0; i < N_in; i++) {
        layer.input[i] = input[i];
    }
    for (size_t i = 0; i < N_out; i++) {
        for (size_t j = 0; j < N_in; j++) {
            layer.weight[i][j] = &graph.create_var(0, name + "_weight_" + std::to_string(i) + "_" + std::to_string(j));
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
