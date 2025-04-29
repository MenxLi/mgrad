use mgrad::nn;
use mgrad::nn::Graph;
use rand::{self, Rng};
use rand_distr::{Distribution, Normal};
use std::fs;
use image;

fn aim_levelset(x: f32, y: f32) -> f32 {
    let rotate = |x: f32, y: f32, theta: f32| -> (f32, f32) {
        let rx = theta.cos() * x - theta.sin() * y;
        let ry = theta.sin() * x + theta.cos() * y;
        (rx, ry)
    };

    let oval = |x: f32, y: f32, a: f32, b: f32, scale: f32| -> f32 {
        x * x / (a * a) + y * y / (b * b) - scale
    };

    let (x1, y1) = rotate(x, y, 0.2);
    let (x2, y2) = rotate(x, y, -0.6);
    f32::min(
        oval(x1 + 1.5, y1 - 1., 1., 2., 1.2),
        oval(x2 - 0.5, y2 + 2., 2., 1., 0.8),
    )
}

fn get_samples<const N: usize>() -> [[nn::fp_t; 3]; N] {
    let mut rng = rand::rng();
    let dist = rand::distr::Uniform::new(-5.0, 5.0).unwrap();

    let mut samples: [[nn::fp_t; 3]; N] = [[0.0; 3]; N];
    for i in 0..N {
        let x = rng.sample(dist);
        let y = rng.sample(dist);
        let z = if aim_levelset(x, y) < 0.0 { 1.0 } else { 0.0 };
        samples[i] = [x.into(), y.into(), z.into()];
    }
    samples
}

fn normal_init(layer: &mut nn::Linear) {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let out_dim = layer.out_dim();
    for i in 0..layer.in_dim() * out_dim {
        layer.weights[i].set_value(normal.sample(&mut rng));
    }
    if let Some(ref mut bias) = layer.bias {
        for i in 0..out_dim {
            bias[i].set_value(0.0);
        }
    }
}

struct Model {
    inputs: Vec<nn::Node>,
    aim: nn::Node,
    predict: nn::Node,
    loss: nn::Node,
}

fn create_model() -> Model {
    let inputs = vec![nn::constant(0), nn::constant(0)];
    let aim = nn::constant(0);

    const W: usize = 8;

    let mut l1 = nn::Linear::new(2, 2 * W).with_bias().with_activation(nn::functional::relu);
    let mut l2 = nn::Linear::new(2 * W, 2 * W).with_bias().with_activation(nn::functional::tanh);
    let mut l3 = nn::Linear::new(2 * W, W).with_bias().with_activation(nn::functional::relu);
    let mut l4 = nn::Linear::new(W, 1).with_activation(nn::functional::sigmoid);

    normal_init(&mut l1);
    normal_init(&mut l2);
    normal_init(&mut l3);
    normal_init(&mut l4);

    let output = l1.forward(&inputs);
    let output = l2.forward(&output);
    let output = l3.forward(&output);
    let output = l4.forward(&output);

    let predict = output.get(0).unwrap().shadow();

    // somehow bce-loss won't work...
    // const EPS: nn::fp_t = 1e-3;
    // let loss = -(
    //     ((&aim + EPS) * (&predict + EPS)).ln() + 
    //     ((1. - &aim + EPS) as nn::Node * (1. - &predict + EPS) as nn::Node).ln()
    // );
    let loss = (&predict - &aim).pow(2);
    Model {
        inputs,
        aim,
        predict,
        loss,
    }
}

fn train_step(
    graph: &mut nn::Graph,
    model: &mut Model,
    n_iter: usize,
){
    const BATCH_SIZE: usize = 32;
    const LR: nn::fp_t = 1e-2;

    let mut batch_loss = 0.0;
    for [x, y, z] in get_samples::<BATCH_SIZE>() {
        model.inputs.get_mut(0).expect("Input 0 not found").set_value(x);
        model.inputs.get_mut(1).expect("Input 1 not found").set_value(y);
        model.aim.set_value(z);

        graph.forward();
        model.loss.backward(1.0);

        batch_loss += model.loss.value;
        graph.apply_grad(-1. * LR);
        graph.zero_grad();
    }

    if (n_iter + 1) % (1e4 as usize) == 0 {
        batch_loss /= BATCH_SIZE as nn::fp_t;
        println!("Loss: {}", batch_loss);
    }
}

fn eval_step(
    graph: &mut nn::Graph,
    model: &mut Model,
    iter: usize,
){
    // create a tmp folder if not exists
    if !std::path::Path::new("tmp").exists() {
        std::fs::create_dir("tmp").unwrap();
    }

    let mut batch_loss = 0.0;
    let mut acc = 0.0;
    for [x, y, z] in get_samples::<32>() {
        model.inputs.get_mut(0).expect("Input 0 not found").set_value(x);
        model.inputs.get_mut(1).expect("Input 1 not found").set_value(y);
        model.aim.set_value(z);

        graph.forward();
        batch_loss += model.loss.value;

        let pred = model.predict.value;
        let pred = if pred > 0.5 { 1.0 } else { 0.0 };
        if pred == z {
            acc += 1.0;
        }
    }
    batch_loss /= 32 as nn::fp_t;
    acc /= 32 as nn::fp_t;

    model.loss.backward(1);
    save_to_file(graph, &format!("tmp/graph-{:06}.gv", iter));

    println!("[iter-{:06}] Acc: {}, Eval Loss: {}", iter, acc, batch_loss);
    graph.zero_grad();

    // sample a 200 * 200 image from [-5, 5]
    let mut img = image::GrayImage::new(200, 200);
    for xi in 0..200 {
        for yi in 0..200 {
            let x = xi as f32 / 200.0 * 10.0 - 5.0;
            let y = yi as f32 / 200.0 * 10.0 - 5.0;
            model.inputs.get_mut(0).expect("Input 0 not found").set_value(x);
            model.inputs.get_mut(1).expect("Input 1 not found").set_value(y);
            graph.forward();
            let pred = model.predict.value;
            let pred = pred.clamp(0., 1.) * 255.0;
            let pred = pred as u8;
            img.put_pixel(xi, yi, image::Luma([pred as u8]));
        }
    }

    // save to file 
    let filename = format!("tmp/graph-{:06}.png", iter);
    img.save(&filename).unwrap();
}

fn save_to_file(g: &nn::Graph, filename: &str) {
    let graph_str = g.to_graphviz();
    let mut file = fs::File::create(filename).unwrap();
    use std::io::Write;
    file.write_all(graph_str.as_bytes()).unwrap();
}


fn main(){
    const N_TOTAL_ITER: usize = 1e4 as usize;

    let mut model = create_model();
    let loss_shadow = model.loss.shadow();
    let mut graph = Graph::from_trace(&loss_shadow).unwrap();

    for i in 0..N_TOTAL_ITER {
        train_step(&mut graph, &mut model, i);
        if i % (1e3 as usize) == 0 {
            eval_step(&mut graph, &mut model, i);
        }
    }
}