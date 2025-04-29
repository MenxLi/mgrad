A minimal automatic scalar differentiation library. 

All code are in a single file (`src/mgrad.rs`) to easily copy-paste into other projects.  
```rust
use mgrad::nn;

fn main() {
    let x = nn::variable(1);
    let y = x.sin() + nn::constant(1);
    let y = (x.pow(2) * y).ln();
    y.backward(1);

    println!("dy/dx at x=1 is: {:?}", x.grad);
}
```

Run with `cargo doc --open` to see the documentation.  
More examples can be found in the `examples` folder.  