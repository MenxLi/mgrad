// Create a new operator
// This example shows how to create a Mean Squared Error (MSE) operator

use mgrad::nn;

struct OpMSE {
    a: nn::Node,
    b: nn::Node,
}

impl nn::OpNode for OpMSE {
    fn inputs(&self) -> Vec<&nn::Node> {
        vec![&self.a, &self.b]
    }

    fn forward_value(&self) -> nn::fp_t {
        (self.a.value - self.b.value).powi(2)
    }

    fn backward(&self, grad: nn::fp_t) {
        let diff = self.a.value - self.b.value;
        self.a.backward(2.0 * diff * grad);
        self.b.backward(-2.0 * diff * grad);
    }

}

fn mse(a: &nn::Node, b: &nn::Node) -> nn::Node {
    let op = OpMSE { a: a.shadow(), b: b.shadow() };    // should not use clone() here
    nn::ops::forward(op)
}

fn main(){
    let a = nn::variable(1.0);
    let b = nn::variable(2.0);

    let c = mse(&a, &b);
    c.backward(1);

    assert_eq!(c.value, 1.0);
    assert_eq!(a.grad, -2.0);
    assert_eq!(b.grad, 2.0);
}