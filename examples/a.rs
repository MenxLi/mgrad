use mgrad::nn;

fn main() {
    let a = nn::variable(2);
    let b = nn::variable(3);

    let c = &a + &b; // This will use the overloaded + operator in Node
    c.backward(1.0); // backward pass with gradient 1.0

    assert_eq!(c.value, 5.0);
    assert_eq!(a.grad, 1.0);
}