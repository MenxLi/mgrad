#include "nn.h"
#include <cmath>

namespace nn {

void OpAdd::forward() {
    output->value = inputs[0]->value + inputs[1]->value;
}
void OpAdd::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad;    // 1 * grad
    if (inputs[1]->requires_grad) inputs[1]->grad += grad;
}

void OpSub::forward() {
    output->value = inputs[0]->value - inputs[1]->value;
}
void OpSub::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad;    // 1 * grad
    if (inputs[1]->requires_grad) inputs[1]->grad -= grad;    // -1 * grad
}

void OpMult::forward() {
    output->value = inputs[0]->value * inputs[1]->value;
}
void OpMult::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * inputs[1]->value;
    if (inputs[1]->requires_grad) inputs[1]->grad += grad * inputs[0]->value;
}

// f(x) = a / b -> ∂f/∂a = 1 / b, ∂f/∂b = -a / b^2
void OpDiv::forward() {
    output->value = inputs[0]->value / inputs[1]->value;
}
void OpDiv::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad / inputs[1]->value;
    if (inputs[1]->requires_grad) inputs[1]->grad -= grad * inputs[0]->value / (inputs[1]->value * inputs[1]->value);
}

// f(x) = a^b -> ∂f/∂a = b * a^(b-1), ∂f/∂b = a^b * log(a)
void OpPow::forward() {
    output->value = pow(inputs[0]->value, inputs[1]->value);
}
void OpPow::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * inputs[1]->value * pow(inputs[0]->value, inputs[1]->value - 1);
    if (inputs[1]->requires_grad) inputs[1]->grad += grad * pow(inputs[0]->value, inputs[1]->value) * log(inputs[0]->value);
}

void OpMax::forward() {
    output->value = std::max(inputs[0]->value, inputs[1]->value);
}
void OpMax::backward(fp_t grad) {
    if (inputs[0]->requires_grad && inputs[0]->value > inputs[1]->value) inputs[0]->grad += grad;
    if (inputs[1]->requires_grad && inputs[1]->value > inputs[0]->value) inputs[1]->grad += grad;
}

void OpMin::forward() {
    output->value = std::min(inputs[0]->value, inputs[1]->value);
}
void OpMin::backward(fp_t grad) {
    if (inputs[0]->requires_grad && inputs[0]->value < inputs[1]->value) inputs[0]->grad += grad;
    if (inputs[1]->requires_grad && inputs[1]->value < inputs[0]->value) inputs[1]->grad += grad;
}

// f(x) = log(a) -> ∂f/∂a = 1 / a
void OpLog::forward() {
    output->value = std::log(inputs[0]->value);
}
void OpLog::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad / inputs[0]->value;
}

void OpMinus::forward() {
    output->value = -inputs[0]->value;
}
void OpMinus::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad -= grad;
}

void OpAbs::forward() {
    output->value = std::abs(inputs[0]->value);
}
void OpAbs::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * (inputs[0]->value > 0 ? 1 : -1);
}

// f(x) = sin(x) -> ∂f/∂x = cos(x)
void OpSin::forward() {
    output->value = sin(inputs[0]->value);
}
void OpSin::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * cos(inputs[0]->value);
}

// f(x) = cos(x) -> ∂f/∂x = -sin(x)
void OpCos::forward() {
    output->value = cos(inputs[0]->value);
}
void OpCos::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad -= grad * sin(inputs[0]->value);
}

void OpRelu::forward() {
    output->value = inputs[0]->value > 0 ? inputs[0]->value : 0;
}
void OpRelu::backward(fp_t grad) {
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * (inputs[0]->value > 0 ? 1 : 0);
}

// f(x) = 1 / (1 + exp(-x)) -> ∂f/∂x = f(x) * (1 - f(x))
void OpSigmoid::forward() {
    output->value = 1 / (1 + exp(-inputs[0]->value));
}
void OpSigmoid::backward(fp_t grad) {
    fp_t s = output->value;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * s * (1 - s);
}

// f(x) = tanh(x) -> ∂f/∂x = 1 - f(x)^2
void OpTanh::forward() {
    output->value = std::tanh(inputs[0]->value);
}
void OpTanh::backward(fp_t grad) {
    fp_t t = output->value;
    if (inputs[0]->requires_grad) inputs[0]->grad += grad * (1 - t * t);
}

}