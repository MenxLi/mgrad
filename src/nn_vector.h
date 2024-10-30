/* Tools for neural network implementation */
#include "nn.h"
#include <array>
#include <memory>

namespace nn {

template <size_t N_in, size_t N_out>
struct LinearLayer {
    Node* input[N_in];
    Node* output[N_out];
    Node* weight[N_out][N_in];
    Node* bias[N_out];
};

template <size_t N_in, size_t N_out>
LinearLayer<N_in, N_out> create_linear_layer(
    Graph& graph, 
    Node* input[N_in],
    const std::string& name = ""
    )
{
    auto layer = LinearLayer<N_in, N_out>();

    for (size_t i = 0; i < N_in; i++) {
        layer.input[i] = input[i];
    }
    for (size_t i = 0; i < N_out; i++) {
        layer.bias[i] = &graph.create_var(0, name + "_bias_" + std::to_string(i));
        for (size_t j = 0; j < N_in; j++) {
            layer.weight[i][j] = &graph.create_var(0, name + "_weight_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }
    for (size_t i = 0; i < N_out; i++) {
        auto& b = *layer.bias[i];
        layer.output[i] = &b;
        for (size_t j = 0; j < N_in; j++) {
            auto& w = *layer.weight[i][j];
            auto& x = *input[j];
            layer.output[i] = &(*layer.output[i] + w * x);
        }
    }
    return layer;
}

enum class ActivationType {
    Relu,
    Sigmoid,
};
template <size_t N>
struct ActivationLayer {
    Node* input[N];
    Node* output[N];
};
template <size_t N>
ActivationLayer<N> create_activation_layer(
    Graph& graph, 
    Node* input[N],
    ActivationType type
    )
{
    auto layer = ActivationLayer<N>();
    for (size_t i = 0; i < N; i++) {
        if (type == ActivationType::Relu) {
            layer.output[i] = graph.relu(input[i]);
        } else if (type == ActivationType::Sigmoid) {
            layer.output[i] = graph.sigmoid(input[i]);
        }
    }
    return layer;
}

}