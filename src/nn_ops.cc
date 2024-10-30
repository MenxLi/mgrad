#include "nn.h"
#include <cmath>

namespace nn {

fp_t OpNode::root_grad() { return output == nullptr ? 1 : output->grad; }

void OpAdd::forward() {
    output->value = inputs[0]->value + inputs[1]->value;
}
void OpAdd::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad;    // 1 * grad
    if (inputs[1]->requires_grad) inputs[1]->grad += grad;
}

void OpSub::forward() {
    output->value = inputs[0]->value - inputs[1]->value;
}
void OpSub::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad;    // 1 * grad
    if (inputs[1]->requires_grad) inputs[1]->grad -= grad;    // -1 * grad
}

void OpMult::forward() {
    output->value = inputs[0]->value * inputs[1]->value;
}
void OpMult::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * inputs[1]->value;
    if (inputs[1]->requires_grad) inputs[1]->grad += grad * inputs[0]->value;
}

// f(x) = a / b -> ∂f/∂a = 1 / b, ∂f/∂b = -a / b^2
void OpDiv::forward() {
    output->value = inputs[0]->value / inputs[1]->value;
}
void OpDiv::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad / inputs[1]->value;
    if (inputs[1]->requires_grad) inputs[1]->grad -= grad * inputs[0]->value / (inputs[1]->value * inputs[1]->value);
}

// power
// f(x) = a^b -> ∂f/∂a = b * a^(b-1), ∂f/∂b = a^b * log(a)
void OpPow::forward() {
    output->value = pow(inputs[0]->value, inputs[1]->value);
}
void OpPow::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * inputs[1]->value * pow(inputs[0]->value, inputs[1]->value - 1);
    if (inputs[1]->requires_grad) inputs[1]->grad += grad * pow(inputs[0]->value, inputs[1]->value) * log(inputs[0]->value);
}

void OpMinus::forward() {
    output->value = -inputs[0]->value;
}
void OpMinus::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad -= grad;
}

// f(x) = 1 / a -> ∂f/∂a = -1 / a^2
void OpInv::forward() {
    output->value = 1 / inputs[0]->value;
}
void OpInv::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad -= grad / (inputs[0]->value * inputs[0]->value);
}

void OpAbs::forward() {
    output->value = std::abs(inputs[0]->value);
}
void OpAbs::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * (inputs[0]->value > 0 ? 1 : -1);
}

void OpRelu::forward() {
    output->value = inputs[0]->value > 0 ? inputs[0]->value : 0;
}
void OpRelu::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * (inputs[0]->value > 0 ? 1 : 0);
}

// f(x) = 1 / (1 + exp(-x)) -> ∂f/∂x = f(x) * (1 - f(x))
void OpSigmoid::forward() {
    output->value = 1 / (1 + exp(-inputs[0]->value));
}
void OpSigmoid::backward() {
    fp_t grad = root_grad(); if (grad == 0) return;
    fp_t s = output->value;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * s * (1 - s);
}

}