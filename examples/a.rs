use mgrad::nn;

fn main() {
    // y = ln(x^2 * (sin(x) + 1))
    let x = nn::variable(1);
    let y = x.sin() + nn::constant(1);
    let y = (x.pow(2) * y).ln();
    y.backward(1);

    // dy/dx should be ~ 2.29341
    println!("The gradient of y=ln(x^2 * (sin(x) + 1)) at x=1 is: {:?}", x.grad);
}