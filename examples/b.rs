use mgrad::nn;

fn main() {
    let x = nn::variable(1);
    let y = nn::variable(2);
    let m = x + y;
    let z = &m.sin() + m.cos();
    z.backward(1);

    let g = nn::Graph::from_trace(&z).unwrap();
    println!("{}", g.to_graphvis())
}