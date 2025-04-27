use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

#[allow(non_camel_case_types)]
type fp_t = f32;

pub trait OpNode {
    fn forward(&self) -> NCell; // should return the output node, no need to set the from field
    fn backward(&self, grad: fp_t); // should invoke backward on the input nodes with proper gradients
}

pub struct NCell(Rc<RefCell<Node>>);
impl NCell {
    pub fn from(node: Node) -> Self {
        NCell(Rc::new(RefCell::new(node)))
    }
    pub fn borrow(&self) -> Ref<Node> {
        self.0.borrow()
    }
    pub fn borrow_mut(&self) -> RefMut<Node> {
        self.0.borrow_mut()
    }
    pub fn clone(r: &NCell) -> Self {
        NCell(Rc::clone(&r.0))
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
        let n = Node::new(self.a.borrow().value + self.b.borrow().value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        let mut a = self.a.borrow_mut();
        let mut b = self.b.borrow_mut();
        a.backward(grad);
        b.backward(grad);
    }
}

struct OpSub {
    a: NCell,
    b: NCell,
}
impl OpNode for OpSub {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.borrow().value - self.b.borrow().value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        let mut a = self.a.borrow_mut();
        let mut b = self.b.borrow_mut();
        a.backward(grad);
        b.backward(-grad);
    }
}

struct OpMul {
    a: NCell,
    b: NCell,
}
impl OpNode for OpMul {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.borrow().value * self.b.borrow().value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        let mut a = self.a.borrow_mut();
        let mut b = self.b.borrow_mut();
        a.backward(grad * b.value);
        b.backward(grad * a.value);
    }
}

struct OpDiv {
    a: NCell,
    b: NCell,
}
impl OpNode for OpDiv {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.borrow().value / self.b.borrow().value);
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        let mut a = self.a.borrow_mut();
        let mut b = self.b.borrow_mut();
        let b_sq = self.b.borrow().value * self.b.borrow().value;
        a.backward(grad / b.value);
        b.backward(-grad * a.value / b_sq);
    }
}

struct OpPow {
    a: NCell,
    b: NCell,
}
impl OpNode for OpPow {
    fn forward(&self) -> NCell {
        let n = Node::new(self.a.borrow().value.powf(self.b.borrow().value));
        NCell::from(n)
    }

    fn backward(&self, grad: fp_t) {
        let mut a = self.a.borrow_mut();
        let mut b = self.b.borrow_mut();
        a.backward(grad * b.value * self.a.borrow().value.powf(b.value - 1.0));
        b.backward(grad * a.value.ln() * self.a.borrow().value.powf(self.b.borrow().value));
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
        o.borrow_mut().from = Some(Box::new(op));
        o
    }

    fn sub_impl(a: &NCell, b: &NCell) -> NCell {
        let op = OpSub {
            a: NCell::clone(a),
            b: NCell::clone(b),
        };
        let o = op.forward();
        o.borrow_mut().from = Some(Box::new(op));
        o
    }

    fn mul_impl(a: &NCell, b: &NCell) -> NCell {
        let op = OpMul {
            a: NCell::clone(a),
            b: NCell::clone(b),
        };
        let o = op.forward();
        o.borrow_mut().from = Some(Box::new(op));
        o
    }

    fn div_impl(a: &NCell, b: &NCell) -> NCell {
        let op = OpDiv {
            a: NCell::clone(a),
            b: NCell::clone(b),
        };
        let o = op.forward();
        o.borrow_mut().from = Some(Box::new(op));
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

impl NCell {
    pub fn pow(self, other: NCell) -> NCell {
        let op = OpPow {
            a: NCell::clone(&self),
            b: NCell::clone(&other),
        };
        let o = op.forward();
        o.borrow_mut().from = Some(Box::new(op));
        o
    }
}

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

    #[test]
    fn test_variable() {
        let x = nn::variable(5.0);
        assert_eq!(x.borrow().value, 5.0);
    }


    #[test]
    fn test_node() {
        let a = NCell::from(Node::new(1.0));
        let b = NCell::from(Node::new(2.0));
        let c = &a + &b;
        c.borrow_mut().backward(1.0); // backward pass with gradient 1.0

        assert_eq!(c.borrow().value, 3.0);
        assert_eq!(a.borrow().grad, 1.0);
        assert_eq!(b.borrow().grad, 1.0);
    }
}
