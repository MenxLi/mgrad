A minimal automatic differentiation library. Just for fun...

Please refer to `examples/a.rs` for usage:
```rust
use mgrad::nn;

fn main() {
    // y = ln(x^2 * (sin(x) + 1))
    let x = nn::variable(1);
    let y = x.sin() + nn::constant(1);
    let y = x.pow(2) * y;
    let y = y.ln();
    y.backward(1);

    // dy/dx should be ~ 2.29341
    println!("The gradient of y=ln(x^2 * (sin(x) + 1)) at x=1 is: {:?}", x.grad);
}
```

Note that `backward()` is called with immutable declared variables to update their gradients.  
This is because some unsafe code is used to avoid `RefCell` overhead.  
Theoretically this is unsafe, but as long as we are not doing parallel computations on the same graph, it should behave correctly.