use std::ops::{Deref, Add, Div, Mul, Sub, Neg};
use std::rc::Rc;

#[allow(non_camel_case_types)]
type fp_t = f32;

pub trait Number {
    fn to_fp(self) -> fp_t;
}
pub trait Nodish {
    fn to_ncell(&self) -> Node;
}
macro_rules!  adapt_num_t{
    ($t:ty) => {
        impl Number for $t {
            fn to_fp(self) -> fp_t { self as fp_t }
        }
        impl Nodish for $t {
            fn to_ncell(&self) -> Node { nn::constant(self.clone()) }
        }
    };
}

adapt_num_t!(i8);
adapt_num_t!(i16);
adapt_num_t!(i32);
adapt_num_t!(i64);
adapt_num_t!(isize);
adapt_num_t!(u8);
adapt_num_t!(u16);
adapt_num_t!(u32);
adapt_num_t!(u64);
adapt_num_t!(usize);
adapt_num_t!(f32);
adapt_num_t!(f64);

pub struct RawNode {
    pub from: Option<Box<dyn OpNode>>,
    pub value: fp_t,
    pub grad: fp_t,
    pub requires_grad: bool,
}

impl RawNode {
    pub fn new(value: fp_t) -> Self {
        RawNode {
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


pub trait OpNode {
    fn forward(&self) -> Node; // should return the output node, no need to set the from field
    fn backward(&self, grad: fp_t); // should invoke backward on the input nodes with proper gradients
}

pub struct Node(Rc<RawNode>);
impl Node {
    pub fn clone(r: &Node) -> Self {
        Node(Rc::clone(&r.0))
    }

    pub fn from(node: RawNode) -> Self {
        Node(Rc::new(node))
    }

    pub fn backward<T: Number>(&self, grad: T) {
        let n = self.get_unsafe_mut();
        n.backward(grad.to_fp());
    }

    pub fn copy(&self) -> Self {
        Node(Rc::clone(&self.0))
    }

    fn get_unsafe_mut(&self) -> &mut RawNode {
        unsafe {
            &mut *(Rc::as_ptr(&self.0) as *mut RawNode)
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
impl Deref for Node {
    type Target = Rc<RawNode>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Nodish for Node {
    fn to_ncell(&self) -> Node {
        Node::clone(self)
    }
}

// ========================== Operations ==========================
struct OpAdd {
    a: Node,
    b: Node,
}
impl OpNode for OpAdd {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value + self.b.value);
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad);
        self.b.backward(grad);
    }
}

struct OpSub {
    a: Node,
    b: Node,
}
impl OpNode for OpSub {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value - self.b.value);
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad);
        self.b.backward(-grad);
    }
}

struct OpMul {
    a: Node,
    b: Node,
}
impl OpNode for OpMul {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value * self.b.value);
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.b.value);
        self.b.backward(grad * self.a.value);
    }
}

struct OpDiv {
    a: Node,
    b: Node,
}
impl OpNode for OpDiv {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value / self.b.value);
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        let b_sq = self.b.value * self.b.value;
        self.a.backward(grad / self.b.value);
        self.b.backward(-grad * self.a.value / b_sq);
    }
}

struct OpPow {
    a: Node,
    b: Node,
}
impl OpNode for OpPow {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value.powf(self.b.value));
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.b.value * self.a.value.powf(self.b.value - 1.0));
        self.b.backward(grad * self.a.value.ln() * self.a.value.powf(self.b.value));
    }
}

struct OpNeg {
    a: Node,
}
impl OpNode for OpNeg {
    fn forward(&self) -> Node {
        let n = RawNode::new(-self.a.value);
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(-grad);
    }
}

struct OpAbs {
    a: Node,
}
impl OpNode for OpAbs {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value.abs());
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.a.value.signum());
    }
}

struct OpLog {
    base: Node,
    val: Node,
}
impl OpNode for OpLog {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.val.value.log(self.base.value));
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.val.backward(grad / (self.val.value * self.base.value.ln()));
        self.base.backward(-grad * self.val.value.ln() / (self.base.value * (self.base.value.ln() * self.base.value.ln())));
    }
}

struct OpSin {
    a: Node,
}
impl OpNode for OpSin {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value.sin());
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.a.value.cos());
    }
}

struct OpCos {
    a: Node,
}
impl OpNode for OpCos {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value.cos());
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(-grad * self.a.value.sin());
    }
}

struct OpTan {
    a: Node,
}
impl OpNode for OpTan {
    fn forward(&self) -> Node {
        let n = RawNode::new(self.a.value.tan());
        Node::from(n)
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad / (self.a.value.cos() * self.a.value.cos()));
    }
}

// These are common operations that can be used to implement for borrowed references and owned values.
struct OpImpl {}
impl OpImpl {
    fn add_impl(a: &Node, b: &Node) -> Node {
        let op = OpAdd {
            a: Node::clone(a),
            b: Node::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn sub_impl(a: &Node, b: &Node) -> Node {
        let op = OpSub {
            a: Node::clone(a),
            b: Node::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn mul_impl(a: &Node, b: &Node) -> Node {
        let op = OpMul {
            a: Node::clone(a),
            b: Node::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn div_impl(a: &Node, b: &Node) -> Node {
        let op = OpDiv {
            a: Node::clone(a),
            b: Node::clone(b),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    fn neg_impl(a: &Node) -> Node {
        let op = OpNeg {
            a: Node::clone(a),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }
}

// ========================== Operator Overloading ==========================
macro_rules! impl_op_2 {
    ($op:ident, $name:ident, $name_impl: ident) => {
        impl $op for Node {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                OpImpl::$name_impl(&self, &other)
            }
        }
        impl $op for &Node {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                OpImpl::$name_impl(self, other)
            }
        }
        impl $op<&Node> for Node {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                OpImpl::$name_impl(&self, other)
            }
        }
        impl $op<Node> for &Node {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                OpImpl::$name_impl(self, &other)
            }
        }
    };
}
macro_rules! impl_op_num_2_t {
    ($t:ty, $op:ident, $name:ident, $name_impl: ident) => {
        impl $op<Node> for $t {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                OpImpl::$name_impl(&nn::constant(self), &other)
            }
        }
        impl $op<&Node> for $t {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                OpImpl::$name_impl(&nn::constant(self), other)
            }
        }
        impl $op<$t> for Node {
            type Output = Node;
            fn $name(self, other: $t) -> Self::Output {
                OpImpl::$name_impl(&self, &nn::constant(other))
            }
        }
        impl $op<$t> for &Node {
            type Output = Node;
            fn $name(self, other: $t) -> Self::Output {
                OpImpl::$name_impl(self, &nn::constant(other))
            }
        }
    };
}
macro_rules! impl_op_num_2 {
    ($op:ident, $name:ident, $name_impl: ident) => {
        impl_op_num_2_t!(i8, $op, $name, $name_impl);
        impl_op_num_2_t!(i16, $op, $name, $name_impl);
        impl_op_num_2_t!(i32, $op, $name, $name_impl);
        impl_op_num_2_t!(i64, $op, $name, $name_impl);
        impl_op_num_2_t!(isize, $op, $name, $name_impl);
        impl_op_num_2_t!(u8, $op, $name, $name_impl);
        impl_op_num_2_t!(u16, $op, $name, $name_impl);
        impl_op_num_2_t!(u32, $op, $name, $name_impl);
        impl_op_num_2_t!(u64, $op, $name, $name_impl);
        impl_op_num_2_t!(usize, $op, $name, $name_impl);
        impl_op_num_2_t!(f32, $op, $name, $name_impl);
        impl_op_num_2_t!(f64, $op, $name, $name_impl);
    }
}

impl_op_2!(Add, add, add_impl);
impl_op_num_2!(Add, add, add_impl);

impl_op_2!(Sub, sub, sub_impl);
impl_op_num_2!(Sub, sub, sub_impl);

impl_op_2!(Mul, mul, mul_impl);
impl_op_num_2!(Mul, mul, mul_impl);

impl_op_2!(Div, div, div_impl);
impl_op_num_2!(Div, div, div_impl);


// single operand operations
impl Neg for Node {
    type Output = Node;
    fn neg(self) -> Self::Output {
        OpImpl::neg_impl(&self)
    }
}
impl Neg for &Node {
    type Output = Node;
    fn neg(self) -> Self::Output {
        OpImpl::neg_impl(self)
    }
}

impl Node {
    pub fn pow(&self, other: &impl Nodish) -> Node {
        let op = OpPow {
            a: Node::clone(&self),
            b: other.to_ncell(),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn abs(&self) -> Node {
        let op = OpAbs {
            a: Node::clone(&self),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn log(&self, base: &impl Nodish) -> Node {
        let op = OpLog {
            base: base.to_ncell(),
            val: Node::clone(&self),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn ln(&self) -> Node {
        self.log(&nn::constant(std::f32::consts::E))
    }

    pub fn sin(&self) -> Node {
        let op = OpSin {
            a: Node::clone(&self),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn cos(&self) -> Node {
        let op = OpCos {
            a: Node::clone(&self),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }

    pub fn tan(&self) -> Node {
        let op = OpTan {
            a: Node::clone(&self),
        };
        let o = op.forward();
        o.get_unsafe_mut().from = Some(Box::new(op));
        o
    }
}

pub mod nn {
    use super::*;

    // bring the Node type into the nn module
    pub use super::Node;

    pub fn variable<T: Number>(value: T) -> Node {
        let n = RawNode::new(value.to_fp());
        Node::from(n)
    }

    pub fn constant<T: Number>(value: T) -> Node {
        let mut n = RawNode::new(value.to_fp());
        n.requires_grad = false;
        Node::from(n)
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

        let mut a = Node::from(RawNode::new(1.0));
        let mut b = Node::from(RawNode::new(2.0));
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

    #[test]
    fn test_const(){
        let a = nn::variable(1);
        let b: Node = &a + 1;
        assert_eq!(b.value, 2.0);

        let c: Node = 1 + a;
        assert_eq!(c.value, 2.0);
    }
}
