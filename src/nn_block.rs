use crate::mgrad::nn;

pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
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
        assert!(t.len() == self.in_dim, "Input dimension mismatch: expected {}, got {}", self.in_dim, t.len());
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
                        out.push(nn::functional::sigmoid(&linear_out[i]));
                    }
                }
                Activation::Tanh => {
                    for i in 0..self.out_dim {
                        out.push(nn::functional::tanh(&linear_out[i]));
                    }
                }
                Activation::ReLU => {
                    for i in 0..self.out_dim {
                        out.push(nn::functional::relu(&linear_out[i]));
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
        let mut graph = nn::Graph::from_trace(&loss).unwrap();

        for _ in 0..n_iter {
            graph.forward();
            loss.backward(1.0);
            graph.apply_grad(-1.0 * lr);
            graph.zero_grad();
        }

        println!("output: {}", output.value);
        println!("aim: {}", aim.value);
        assert!((output.value - aim.value).abs() < 5e-2);
    }

    #[test]
    fn test_complex2(){

        let a = nn::variable(-4);
        let b = nn::variable(2);

        let c = &a + &b;
        let d = &a * &b + &b.pow(3);

        let c = &c + (&c + 1);
        let c = &c + 1 + c + (-&a);

        let d = &d + &d * 2 + nn::functional::relu(&(&b + &a));
        let d = &d + 3 * &d + nn::functional::relu(&(&b -&a));

        let e: nn::Node = c - d;
        let f = e.pow(2);

        let g = &f / 2;
        let g1: nn::Node = &g + 10 / f;

        g1.backward(1);

        assert_close(a.grad, 138.8338, 1e-3);
        assert_close(b.grad, 645.5773, 1e-3);

    }
}