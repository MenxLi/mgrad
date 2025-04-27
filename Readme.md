A minimal automatic differentiation library. Just for fun...

Please refer to `examples/a.rs` for usage:
```rust
use mgrad::nn;

fn main() {
    let x = nn::variable(1);
    let mut y: nn::Node = x.pow(&2) * (x.sin() + 1);
    y = y.ln();
    y.backward(1);

    // y = log(x^2 * (sin(x) + 1))
    // should be ~2.2934
    println!("The gradient of y=ln(x^2 * (sin(x) + 1)) at x=1 is: {:?}", x.grad);
}
```