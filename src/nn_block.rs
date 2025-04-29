use crate::mgrad::nn;

pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
}

mod activation {
    use crate::mgrad::nn;
    use crate::mgrad::OpNode;
    use crate::nn::fp_t;


    struct OpSigmoid (nn::Node);
    struct OpTanh (nn::Node);
    struct OpReLU (nn::Node);

    impl OpNode for OpSigmoid {
        fn inputs(&self) -> Vec<&nn::Node> {
            vec![&self.0]
        }
        fn forward_value(&self) -> fp_t {
            const E: f32 = std::f32::consts::E;
            1. / (1. + E.powf(-self.0.value))
        }
        fn backward(&self, grad: fp_t) {
            let sigmoid = self.forward_value();
            self.0.backward(grad * sigmoid * (1. - sigmoid));
        }
    }

    impl OpNode for OpTanh {
        fn inputs(&self) -> Vec<&nn::Node> {
            vec![&self.0]
        }
        fn forward_value(&self) -> fp_t {
            self.0.value.tanh()
        }
        fn backward(&self, grad: fp_t) {
            let tanh = self.forward_value();
            self.0.backward(grad * (1. - tanh * tanh));
        }
    }

    impl OpNode for OpReLU {
        fn inputs(&self) -> Vec<&nn::Node> {
            vec![&self.0]
        }
        fn forward_value(&self) -> fp_t {
            self.0.value.max(0.)
        }
        fn backward(&self, grad: fp_t) {
            if self.0.value >= 0. {
                self.0.backward(grad);
            } else {
                self.0.backward(0.0);
            }
        }
    }

    pub fn sigmoid(x: &nn::Node) -> nn::Node {
        let op = OpSigmoid(x.shadow());
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn tanh(x: &nn::Node) -> nn::Node {
        let op = OpTanh(x.shadow());
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn relu(x: &nn::Node) -> nn::Node {
        let op = OpReLU(x.shadow());
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }
}

pub struct LinearLayer {
    pub weights: Vec<nn::Node>, 
    pub bias: Vec<nn::Node>,
    pub activation: Option<Activation>,
    in_dim: usize,
    out_dim: usize,
}

impl LinearLayer {
    pub fn new(in_dim: usize, out_dim: usize, act: &'static str) -> Self {
        let weights = vec![nn::variable(0); in_dim * out_dim];
        let bias = vec![nn::variable(0); out_dim];
        let activation = match act {
            "sigmoid" => Some(Activation::Sigmoid),
            "tanh" => Some(Activation::Tanh),
            "relu" => Some(Activation::ReLU),
            "" => None,
            _ => panic!("Unknown activation function: {}", act),
        };
        LinearLayer {
            weights,
            bias,
            activation, 
            in_dim,
            out_dim,
        }
    }

    pub fn forward(&mut self, t: &Vec<nn::Node>) -> Vec<nn::Node> {
        let mut linear_out = Vec::with_capacity(self.out_dim);
        for i in 0..self.out_dim {
            linear_out.push(self.bias[i].shadow());
            for j in 0..self.in_dim {
                linear_out[i] = &linear_out[i] + &t[j] * &self.weights[i * self.in_dim + j];
            }
        }
        let mut out = Vec::with_capacity(self.out_dim);
        if let Some(act) = &self.activation {
            match act {
                Activation::Sigmoid => {
                    for i in 0..self.out_dim {
                        out.push(activation::sigmoid(&linear_out[i]));
                    }
                }
                Activation::Tanh => {
                    for i in 0..self.out_dim {
                        out.push(activation::tanh(&linear_out[i]));
                    }
                }
                Activation::ReLU => {
                    for i in 0..self.out_dim {
                        out.push(activation::relu(&linear_out[i]));
                    }
                }
            }
        }
        else {
            out = linear_out;
        }
        out
    }
}

pub fn linear(in_dim: usize, out_dim: usize, act: &'static str) -> LinearLayer {
    LinearLayer::new(in_dim, out_dim, act)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mgrad::nn;

    fn assert_close(a: nn::fp_t, b: nn::fp_t, epsilon: nn::fp_t) {
        assert!((a - b).abs() < epsilon, "assertion failed: {} != {}", a, b);
    }

    #[test]
    fn test_linear_layer() {
        let mut layer = linear(2, 3, "sigmoid");
        let input = vec![nn::variable(1.0), nn::variable(2.0)];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_linear_reg() {
        let mut layer = linear(2, 1, "sigmoid");
        let input = vec![nn::constant(1.0), nn::constant(2.0)];
        let output = layer.forward(&input).get(0).unwrap().shadow();

        for i in &mut layer.weights.iter() {
            i.get_unsafe_mut().value=1.;
        }
        for i in &mut layer.bias.iter() {
            i.get_unsafe_mut().value=1.;
        }

        let aim = nn::constant(1.0);
        let loss = (&output - &aim).pow(2);
        let n_iter = 1000;
        let lr = 1e-3;
        let mut graph = nn::Graph::from_trace(&output).unwrap();

        for _ in 0..n_iter {
            graph.forward();
            loss.backward(1.0);
            graph.apply_grad(-1.0 * lr);
        }

        println!("output: {}", output.value);
        println!("aim: {}", aim.value);
        assert!((output.value - aim.value).abs() < 1e-2);
    }

    #[test]
    fn test_complex2(){

        let a = nn::variable(-4);
        let b = nn::variable(2);

        let c = &a + &b;
        let d = &a * &b + &b.pow(3);

        let c = &c + (&c + 1);
        let c = &c + 1 + c + (-&a);

        let d = &d + &d * 2 + activation::relu(&(&b + &a));
        let d = &d + 3 * &d + activation::relu(&(&b -&a));

        let e: nn::Node = c - d;
        let f = e.pow(2);

        let g = &f / 2;
        let g1: nn::Node = &g + 10 / f;

        g1.backward(1);

        assert_close(a.grad, 138.8338, 1e-3);
        assert_close(b.grad, 645.5773, 1e-3);

    }
}