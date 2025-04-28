use mgrad::nn;

fn main() {
    let mut x = nn::variable(1);
    let mut y = nn::variable(2);
    let m = &x + &y;
    let z = &m.sin() + &m.cos();
    z.abs().backward(1);

    // Capture the graph up to z, 
    // and print it in graphviz format
    let mut g = nn::Graph::from_trace(&z).unwrap();
    println!("{}", g.to_graphvis());

    // Perform gradient descent, 
    // follow the graph to re-calculate z
    let z_value_record = z.value;
    let step = 1e-3;
    x.set_value(x.value - step * x.grad);
    y.set_value(y.value - step * y.grad);
    g.forward();
    assert!(z.value.abs() < z_value_record.abs());
}