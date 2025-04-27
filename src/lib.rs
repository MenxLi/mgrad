// use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Add, Deref, Div, Mul, Sub, Neg};
use std::rc::Rc;

#[allow(non_camel_case_types)]
type fp_t = f32;

pub trait Number { fn to_fp(self) -> fp_t; }
impl Number for i8    { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for i16   { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for i32   { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for i64   { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for isize { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for u8    { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for u16   { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for u32   { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for u64   { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for usize { fn to_fp(self) -> fp_t { self as fp_t } }
impl Number for fp_t   { fn to_fp(self) -> fp_t { self } }
impl Number for f64   { fn to_fp(self) -> fp_t { self as fp_t } }


pub trait OpNode {
    fn forward(&self) -> NCell; // should return the output node, no need to set the from field
    fn backward(&self, grad: fp_t); // should invoke backward on the input nodes with proper gradients
}

pub struct NCell(Rc<Node>);
impl NCell {
    pub fn clone(r: &NCell) -> Self {
        NCell(Rc::clone(&r.0))
    }

    pub fn from(node: Node) -> Self {
        NCell(Rc::new(node))
    }

    pub fn backward(&self, grad: fp_t) {
        let n = self.get_unsafe_mut();
        n.backward(grad);
    }

    pub fn copy(&self) -> Self {
        NCell(Rc::clone(&self.0))
    }

    fn get_unsafe_mut(&self) -> &mut Node {
        unsafe {
            &mut *(Rc::as_ptr(&self.0) as *mut Node)
        }
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        let n = self.get_unsafe_mut();
        n.requires_grad = requires_grad;
    }
    pub fn set_value<T:Number>(&mut self, value: T) {
        let n = self.get_unsafe_mut();
        n.value = value.to_fp();
    }
    pub fn set_grad<T:Number>(&mut self, grad: T) {
        let n = self.get_unsafe_mut();
        n.grad = grad.to_fp();
    }
    pub fn zero_grad(&mut self) {
        self.set_grad(0.0);
    }

}
impl Deref for NCell {
    type Target = Rc<Node>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Node {
    pub from: Option<Box<dyn OpNode>>,
    pub value: fp_t,
    pub grad: fp_t,
    pub requires_grad: bool,
}

impl Node {
    pub fn new(value: fp_t) -> Self {
        Node {
            from: None,
            value: value,
            grad: 0.0,
            requires_grad: true,
        }
    }

    pub fn backward(&mut self, grad: fp_t) {
        if !self.requires_grad {
            return;
        }

        self.grad += grad;
        if let Some(ref op) = self.from {
            op.backward(self.grad);
            self.grad = 0.0; // reset the gradient after backpropagation
        }
    }
}

// ========================== Operations ==========================
struct OpAdd {
    a: NCell,
    b: NCell,
}
impl OpNode for OpAdd {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.value + self.b.value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad);
        self.b.backward(grad);
    }
}

struct OpSub {
    a: NCell,
    b: NCell,
}
impl OpNode for OpSub {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.value - self.b.value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad);
        self.b.backward(-grad);
    }
}

struct OpMul {
    a: NCell,
    b: NCell,
}
impl OpNode for OpMul {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.value * self.b.value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.b.value);
        self.b.backward(grad * self.a.value);
    }
}

struct OpDiv {
    a: NCell,
    b: NCell,
}
impl OpNode for OpDiv {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.value / self.b.value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        let b_sq = self.b.value * self.b.value;
        self.a.backward(grad / self.b.value);
        self.b.backward(-grad * self.a.value / b_sq);
    }
}

struct OpPow {
    a: NCell,
    b: NCell,
}
impl OpNode for OpPow {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.value.powf(self.b.value));
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.b.value * self.a.value.powf(self.b.value - 1.0));
        self.b.backward(grad * self.a.value.ln() * self.a.value.powf(self.b.value));
    }
}

struct OpNeg {
    a: NCell,
}
impl OpNode for OpNeg {
    fn forward(&self) -> NCell {
        let n = Node::new(-self.a.value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(-grad);
    }
}

struct OpAbs {
    a: NCell,
}
impl OpNode for OpAbs {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.value.abs());
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.a.value.signum());
    }
}

struct OpLog {
    base: NCell,
    val: NCell,
}
impl OpNode for OpLog {
    fn forward(&self) -> NCell {
        let n = Node::new(self.val.value.log(self.base.value));
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.val.backward(grad / (self.val.value * self.base.value.ln()));
        self.base.backward(-grad * self.val.value.ln() / (self.base.value * (self.base.value.ln() * self.base.value.ln())));
    }
}

// These are common operations that can be used to implement for borrowed references and owned values.
struct OpImpl {}
impl OpImpl {
    fn add_impl(a: &NCell, b: &NCell) -> NCell {
        let op = OpAdd {
            a: NCell::clone(a),
            b: NCell::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn sub_impl(a: &NCell, b: &NCell) -> NCell {
        let op = OpSub {
            a: NCell::clone(a),
            b: NCell::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn mul_impl(a: &NCell, b: &NCell) -> NCell {
        let op = OpMul {
            a: NCell::clone(a),
            b: NCell::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn div_impl(a: &NCell, b: &NCell) -> NCell {
        let op = OpDiv {
            a: NCell::clone(a),
            b: NCell::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn neg_impl(a: &NCell) -> NCell {
        let op = OpNeg {
            a: NCell::clone(a),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }
}

// ========================== Operator Overloading ==========================
impl Add for NCell {
    type Output = NCell;
    fn add(self, other: NCell) -> Self::Output {
        OpImpl::add_impl(&self, &other)
    }
}
impl Add for &NCell {
    type Output = NCell;
    fn add(self, other: &NCell) -> Self::Output {
        OpImpl::add_impl(self, other)
    }
}

impl Sub for NCell {
    type Output = NCell;
    fn sub(self, other: NCell) -> Self::Output {
        OpImpl::sub_impl(&self, &other)
    }
}
impl Sub for &NCell {
    type Output = NCell;
    fn sub(self, other: &NCell) -> Self::Output {
        OpImpl::sub_impl(self, other)
    }
}

impl Mul for NCell {
    type Output = NCell;
    fn mul(self, other: NCell) -> Self::Output {
        OpImpl::mul_impl(&self, &other)
    }
}
impl Mul for &NCell {
    type Output = NCell;
    fn mul(self, other: &NCell) -> Self::Output {
        OpImpl::mul_impl(self, other)
    }
}

impl Div for NCell {
    type Output = NCell;
    fn div(self, other: NCell) -> Self::Output {
        OpImpl::div_impl(&self, &other)
    }
}
impl Div for &NCell {
    type Output = NCell;
    fn div(self, other: &NCell) -> Self::Output {
        OpImpl::div_impl(self, other)
    }
}

impl Neg for NCell {
    type Output = NCell;
    fn neg(self) -> Self::Output {
        OpImpl::neg_impl(&self)
    }
}

impl NCell {
    pub fn pow(&self, other: &NCell) -> NCell {
        let op = OpPow {
            a: NCell::clone(&self),
            b: NCell::clone(&other),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn abs(&self) -> NCell {
        let op = OpAbs {
            a: NCell::clone(&self),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn log(&self, base: &NCell) -> NCell {
        let op = OpLog {
            base: NCell::clone(&base),
            val: NCell::clone(&self),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn ln(&self) -> NCell {
        self.log(&nn::constant(std::f32::consts::E))
    }
}

pub mod nn {
    use super::*;

    pub fn variable<T: Number>(value: T) -> NCell {
        let n = Node::new(value.to_fp());
        NCell::from(n)
    }

    pub fn constant<T: Number>(value: T) -> NCell {
        let mut n = Node::new(value.to_fp());
        n.requires_grad = false;
        NCell::from(n)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn assert_close(a: fp_t, b: fp_t, epsilon: fp_t) {
        assert!((a - b).abs() < epsilon, "assertion failed: {} != {}", a, b);
    }

    #[test]
    fn test_variable() {
        let x = nn::variable(5.0);
        assert_eq!(x.value, 5.0);
        assert_eq!(x.grad, 0.0);
        assert_eq!(x.requires_grad, true);
    }

    #[test]
    fn test_constant() {
        let x = nn::constant(5.0);
        assert_eq!(x.value, 5.0);
        assert_eq!(x.requires_grad, false);
    }


    #[test]
    fn test_simple() {

        let mut a = NCell::from(Node::new(1.0));
        let mut b = NCell::from(Node::new(2.0));
        let c = &a + &b;
        c.backward(1.0);

        assert_eq!(c.value, 3.0);
        assert_eq!(a.grad, 1.0);
        assert_eq!(b.grad, 1.0);
        a.zero_grad();
        b.zero_grad();

        let d = &a * &b;
        d.backward(2.0);    // backward pass with gradient 2.0
        assert_eq!(d.value, 2.0);
        assert_eq!(a.grad, 4.0);
        assert_eq!(b.grad, 2.0);
        a.zero_grad();
        b.zero_grad();

        let e = &a / &b;
        e.backward(1.0);
        assert_eq!(e.value, 0.5);
        assert_eq!(a.grad, 0.5);
        assert_eq!(b.grad, -0.25);
        a.zero_grad();
        b.zero_grad();

        let f = &a.pow(&b);
        f.backward(1.0);
        assert_eq!(f.value, 1.0);
        assert_eq!(a.grad, 2.0);
        assert_eq!(b.grad, 0.0);
        a.zero_grad();
        b.zero_grad();

        let g = &b.abs();
        g.backward(1.0);
        assert_eq!(g.value, 2.0);
        assert_eq!(b.grad, 1.0);
        b.zero_grad();

        let h = &a.log(&b);
        h.backward(1.0);
        assert_eq!(h.value, 0.0);
        assert_close(a.grad, 1.44269, 1e-3);
        assert_close(b.grad, -0.0, 1e-3);

    }
}
