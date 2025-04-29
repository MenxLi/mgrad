use crate::mgrad::nn;

pub struct LinearLayer {
    pub weights: Vec<nn::Node>, 
    pub bias: Option<Vec<nn::Node>>,
    pub activation: Option<fn(&nn::Node) -> nn::Node>,
    in_dim: usize,
    out_dim: usize,
}

impl LinearLayer {
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

impl LinearLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = vec![nn::variable(0); in_dim * out_dim];
        // let bias = vec![nn::variable(0); out_dim];
        LinearLayer {
            weights,
            bias: None,
            activation: None, 
            in_dim,
            out_dim,
        }
    }

    pub fn with_activation(self, act: fn(&nn::Node) -> nn::Node) -> Self {
        LinearLayer {
            activation: Some(act),
            ..self
        }
    }

    pub fn with_bias(self) -> Self {
        let bias = Some(vec![nn::variable(0); self.out_dim]);
        LinearLayer {
            bias,
            ..self
        }
    }

    pub fn forward(&mut self, t: &Vec<nn::Node>) -> Vec<nn::Node> {
        assert!(t.len() == self.in_dim, "Input dimension mismatch: expected {}, got {}", self.in_dim, t.len());
        let mut linear_out = Vec::with_capacity(self.out_dim);
        for i in 0..self.out_dim {
            linear_out.push(&t[0] * &self.weights[i * self.in_dim + 0]);
            if self.in_dim > 0 {
                for j in 1..self.in_dim {
                    linear_out[i] = &linear_out[i] + &t[j] * &self.weights[i * self.in_dim + j];
                }
            }
            if let Some(b) = &self.bias {
                linear_out[i] = &linear_out[i] + &b[i];
            }
        }
        let mut out = Vec::with_capacity(self.out_dim);
        if let Some(act) = &self.activation {
            for i in 0..self.out_dim {
                out.push(act(&linear_out[i]));
            }
        }
        else {
            out = linear_out;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mgrad::nn;

    #[test]
    fn test_linear_layer() {
        let mut layer = LinearLayer::new(2, 3).with_bias().with_activation(nn::functional::sigmoid);
        let input = vec![nn::variable(1.0), nn::variable(2.0)];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_linear_reg() {
        let mut layer = LinearLayer::new(2, 1).with_bias().with_activation(nn::functional::sigmoid);
        let input = vec![nn::constant(1.0), nn::constant(2.0)];
        let output = layer.forward(&input).get(0).unwrap().shadow();

        for i in &mut layer.weights.iter() {
            i.get_unsafe_mut().value=1.;
        }
        if let Some(b) = &mut layer.bias {
            for i in b.iter() {
                i.get_unsafe_mut().value=1.;
            }
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

}