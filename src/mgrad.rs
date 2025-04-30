/// Author: Li, Mengxun <mengxunli@whu.edu.cn>
/// Repository: https://github.com/menxli/mgrad
/// File created: 2025-04-26

use std::ops::{Deref, Add, Div, Mul, Sub, Neg};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::fmt;

#[allow(non_camel_case_types)]
pub type fp_t = f32;

pub trait Number {
    fn to_fp(self) -> fp_t;
}
pub trait Nodish {
    fn to_node(&self) -> Node;
}
macro_rules!  adapt_num_t{
    ($t:ty) => {
        impl Number for $t {
            fn to_fp(self) -> fp_t { self as fp_t }
        }
        impl Nodish for $t {
            fn to_node(&self) -> Node { nn::constant(self.clone()) }
        }
        impl Nodish for &$t {
            fn to_node(&self) -> Node { nn::constant((*self).clone()) }
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
            value,
            from: None,
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

    fn address(&self) -> usize {
        self as *const Self as *const () as usize
    }
}


pub trait OpNode {
    fn name(&self) -> &'static str {
        let full_name = std::any::type_name::<Self>();
        full_name.rsplit_once("::").map_or(full_name, |(_, name)| name)
    }

    /// for backtracking, we need to know the input nodes
    fn inputs(&self) -> Vec<&Node>;

    /// only calculate the value of the output node
    /// should call `forward_op` to create the node
    fn forward_value(&self) -> fp_t; 

    /// should invoke backward on the input nodes with proper gradients
    fn backward(&self, grad: fp_t); 

    fn address(&self) -> usize {
        self as *const Self as *const () as usize
    }
}

/// Basic building block of the computation graph, referring to the underlying `RawNode`.
pub struct Node(Rc<RawNode>);
impl Node {

    pub fn raw(&self) -> &RawNode {
        self.0.as_ref()
    }

    fn clone(r: &Node) -> Self {
        Node(Rc::clone(&r.0))
    }

    pub fn from(node: RawNode) -> Self {
        Node(Rc::new(node))
    }

    /// Update the gradient for all leaf nodes in the graph.  
    /// Note: While this function takes immutable reference, 
    /// it will mutate the internal state of the node of the same graph.
    pub fn backward<T: Number>(&self, grad: T) {
        let n = self.get_unsafe_mut();
        n.backward(grad.to_fp());
    }

    /// Get a shadow of the node, which is a clone of the node referring to the same underlying data.  
    /// We can get mutable reference for each shadow node to mutate the underlying data.  
    /// Thus, should keep as less shadow nodes as possible.  
    pub fn shadow(&self) -> Self {
        Node(Rc::clone(&self.0))
    }

    /// Get a mutable reference to the underlying data of the node.  
    /// Allows mutating the data of the `Rc<RawNode>` directly, to avoid the `RefCell` overhead.
    pub fn get_unsafe_mut(&self) -> &mut RawNode {
        unsafe {
            &mut *(Rc::as_ptr(&self.0) as *mut RawNode)
        }
    }

    /// Check if the node is a leaf node.
    /// A leaf node is a node that is not the result of an operation, 
    /// and will have gradients computed for it.
    pub fn is_leaf(&self) -> bool {
        self.from.is_none()
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
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.raw();
        f.debug_struct("Node")
            .field("value", &n.value)
            .field("grad", &n.grad)
            .field("requires_grad", &n.requires_grad)
            .field("op_fn", &n.from.as_ref().map(|op| op.name()))
            .field("ptr", &n.address())
            .finish()
    }
}
impl Deref for Node {
    type Target = Rc<RawNode>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Clone for Node {
    fn clone(&self) -> Self {
        Node::from(
            RawNode { 
                from: None, 
                value: self.value, 
                grad: self.grad, 
                requires_grad: self.requires_grad,
            }
        )
    }
}
impl Nodish for Node {
    fn to_node(&self) -> Node {
        Node::clone(self)
    }
}
impl Nodish for &Node {
    fn to_node(&self) -> Node {
        Node::clone(self)
    }
}

// ========================== Operations ==========================
struct OpAdd {
    a: Node,
    b: Node,
}
impl OpNode for OpAdd {
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a, &self.b]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value + self.b.value
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
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a, &self.b]
    }
    
    fn forward_value(&self) -> fp_t {
        self.a.value - self.b.value
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
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a, &self.b]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value * self.b.value
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
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a, &self.b]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value / self.b.value
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
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a, &self.b]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value.powf(self.b.value)
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
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a]
    }

    fn forward_value(&self) -> fp_t {
        -self.a.value
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(-grad);
    }
}

struct OpAbs {
    a: Node,
}
impl OpNode for OpAbs {
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value.abs()
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
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.base, &self.val]
    }

    fn forward_value(&self) -> fp_t {
        self.val.value.log(self.base.value)
    }

    fn backward(&self, grad: fp_t) {
        self.val.backward(grad / (self.val.value * self.base.value.ln()));
        self.base.backward(-grad * self.val.value.ln() / (self.base.value * (self.base.value.ln() * self.base.value.ln())));
    }
}

struct OpLn {
    a: Node,
}
impl OpNode for OpLn {
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value.ln()
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad / self.a.value);
    }
}

struct OpSin {
    a: Node,
}
impl OpNode for OpSin {
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value.sin()
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad * self.a.value.cos());
    }
}

struct OpCos {
    a: Node,
}
impl OpNode for OpCos {
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value.cos()
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(-grad * self.a.value.sin());
    }
}

struct OpTan {
    a: Node,
}
impl OpNode for OpTan {
    fn inputs(&self) -> Vec<&Node> {
        vec![&self.a]
    }

    fn forward_value(&self) -> fp_t {
        self.a.value.tan()
    }

    fn backward(&self, grad: fp_t) {
        self.a.backward(grad / (self.a.value.cos() * self.a.value.cos()));
    }
}

pub fn forward_op<T: OpNode + 'static>(op: T) -> Node {
    let mut n = RawNode::new(op.forward_value());
    n.from = Some(Box::new(op));
    Node::from(n)
}

// ========================== Common Operations ==========================
// These are primitive operations, 
// many of them can be used to implement for 
// borrowed references and owned values on operator overloading.
pub fn add(a: &Node, b: &Node) -> Node {
    let op = OpAdd {
        a: Node::clone(a),
        b: Node::clone(b),
    };
    forward_op(op)
}

pub fn sub(a: &Node, b: &Node) -> Node {
    let op = OpSub {
        a: Node::clone(a),
        b: Node::clone(b),
    };
    forward_op(op)
}

pub fn mul(a: &Node, b: &Node) -> Node {
    let op = OpMul {
        a: Node::clone(a),
        b: Node::clone(b),
    };
    forward_op(op)
}

pub fn div(a: &Node, b: &Node) -> Node {
    let op = OpDiv {
        a: Node::clone(a),
        b: Node::clone(b),
    };
    forward_op(op)
}

pub fn neg(a: &Node) -> Node {
    let op = OpNeg {
        a: Node::clone(a),
    };
    forward_op(op)
}

pub fn pow(a: &Node, b: &Node) -> Node {
    let op = OpPow {
        a: Node::clone(a),
        b: Node::clone(b),
    };
    forward_op(op)
}

pub fn abs(a: &Node) -> Node {
    let op = OpAbs {
        a: Node::clone(a),
    };
    forward_op(op)
}

pub fn log(a: &Node, base: impl Nodish) -> Node {
    let op = OpLog {
        base: base.to_node(),
        val: Node::clone(a),
    };
    forward_op(op)
}

pub fn ln(a: &Node) -> Node {
    let op = OpLn {
        a: Node::clone(a),
    };
    forward_op(op)
}

pub fn sin(a: &Node) -> Node {
    let op = OpSin {
        a: Node::clone(a),
    };
    forward_op(op)
}

pub fn cos(a: &Node) -> Node {
    let op = OpCos {
        a: Node::clone(a),
    };
    forward_op(op)
}

pub fn tan(a: &Node) -> Node {
    let op = OpTan {
        a: Node::clone(a),
    };
    forward_op(op)
}

// ========================== Operator Overloading ==========================
macro_rules! impl_op_2 {
    ($op:ident, $name:ident, $name_impl: ident) => {
        impl $op for Node {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                $name_impl(&self, &other)
            }
        }
        impl $op for &Node {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                $name_impl(self, other)
            }
        }
        impl $op<&Node> for Node {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                $name_impl(&self, other)
            }
        }
        impl $op<Node> for &Node {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                $name_impl(self, &other)
            }
        }
    };
}
macro_rules! impl_op_num_2_t {
    ($t:ty, $op:ident, $name:ident, $name_impl: ident) => {
        impl $op<Node> for $t {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                $name_impl(&nn::constant(self), &other)
            }
        }
        impl $op<&Node> for $t {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                $name_impl(&nn::constant(self), other)
            }
        }
        impl $op<Node> for &$t {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                $name_impl(&nn::constant(*self), &other)
            }
        }
        impl $op<&Node> for &$t {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                $name_impl(&nn::constant(*self), other)
            }
        }
        impl $op<$t> for Node {
            type Output = Node;
            fn $name(self, other: $t) -> Self::Output {
                $name_impl(&self, &nn::constant(other))
            }
        }
        impl $op<&$t> for Node {
            type Output = Node;
            fn $name(self, other: &$t) -> Self::Output {
                $name_impl(&self, &nn::constant(*other))
            }
        }
        impl $op<$t> for &Node {
            type Output = Node;
            fn $name(self, other: $t) -> Self::Output {
                $name_impl(self, &nn::constant(other))
            }
        }
        impl $op<&$t> for &Node {
            type Output = Node;
            fn $name(self, other: &$t) -> Self::Output {
                $name_impl(self, &nn::constant(*other))
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

impl_op_2!(Add, add, add);
impl_op_num_2!(Add, add, add);

impl_op_2!(Sub, sub, sub);
impl_op_num_2!(Sub, sub, sub);

impl_op_2!(Mul, mul, mul);
impl_op_num_2!(Mul, mul, mul);

impl_op_2!(Div, div, div);
impl_op_num_2!(Div, div, div);


// single operand operations
impl Neg for Node {
    type Output = Node;
    fn neg(self) -> Self::Output {
        neg(&self)
    }
}
impl Neg for &Node {
    type Output = Node;
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl Node {
    pub fn pow(&self, other: impl Nodish) -> Node {
        pow(self, &other.to_node())
    }

    pub fn abs(&self) -> Node {
        abs(self)
    }

    pub fn log(&self, base: impl Nodish) -> Node {
        log(self, base)
    }

    pub fn ln(&self) -> Node {
        ln(self)
    }

    pub fn sin(&self) -> Node {
        sin(self)
    }

    pub fn cos(&self) -> Node {
        cos(self)
    }

    pub fn tan(&self) -> Node {
        tan(self)
    }
}


// ============================ Activation Functions ==========================

/// Some higher-level functions (for common activation functions)
pub mod functional {
    use super::nn;
    use super::OpNode;
    use super::fp_t;
    use super::forward_op;

    struct OpSigmoid (nn::Node);
    struct OpTanh (nn::Node);
    struct OpReLU (nn::Node);

    impl OpNode for OpSigmoid {
        fn inputs(&self) -> Vec<&nn::Node> {
            vec![&self.0]
        }
        fn forward_value(&self) -> fp_t {
            const E: nn::fp_t = std::f32::consts::E as nn::fp_t;
            1. / (1. + E.powf(-self.0.value))
        }
        fn backward(&self, grad: fp_t) {
            let sigmoid = self.forward_value();
            self.0.backward(grad * sigmoid * (1. - sigmoid));
        }
    }

    impl OpNode for OpTanh {
        fn inputs(&self) -> Vec<&nn::Node> {
            vec![&self.0]
        }
        fn forward_value(&self) -> fp_t {
            self.0.value.tanh()
        }
        fn backward(&self, grad: fp_t) {
            let tanh = self.forward_value();
            self.0.backward(grad * (1. - tanh * tanh));
        }
    }

    impl OpNode for OpReLU {
        fn inputs(&self) -> Vec<&nn::Node> {
            vec![&self.0]
        }
        fn forward_value(&self) -> fp_t {
            self.0.value.max(0.)
        }
        fn backward(&self, grad: fp_t) {
            if self.0.value >= 0. {
                self.0.backward(grad);
            } else {
                self.0.backward(0.0);
            }
        }
    }

    pub fn sigmoid(x: &nn::Node) -> nn::Node {
        let op = OpSigmoid(x.shadow());
        forward_op(op)
    }

    pub fn tanh(x: &nn::Node) -> nn::Node {
        let op = OpTanh(x.shadow());
        forward_op(op)
    }

    pub fn relu(x: &nn::Node) -> nn::Node {
        let op = OpReLU(x.shadow());
        forward_op(op)
    }

}

// ============================ Linear layer ==========================

/// Fully connected linear layer.
/// 
/// Example:
/// ```rust
/// use mgrad::nn;
/// use mgrad::nn::Linear;
/// 
/// let mut l = Linear::new(2, 3).with_bias().with_activation(nn::functional::relu);
/// let x = nn::constant(0.5);
/// let y = nn::constant(1.0);
/// let output = l.forward(&vec![x, y]);
/// assert_eq!(output.len(), 3);
/// ```
/// 
pub struct Linear {
    pub weights: Vec<nn::Node>, 
    pub bias: Option<Vec<nn::Node>>,
    pub activation: Option<fn(&nn::Node) -> nn::Node>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = vec![nn::variable(0); in_dim * out_dim];
        // let bias = vec![nn::variable(0); out_dim];
        Linear {
            weights,
            bias: None,
            activation: None, 
            in_dim,
            out_dim,
        }
    }

    pub fn with_activation(self, act: fn(&nn::Node) -> nn::Node) -> Self {
        Linear {
            activation: Some(act),
            ..self
        }
    }

    pub fn with_bias(self) -> Self {
        let bias = Some(vec![nn::variable(0); self.out_dim]);
        Linear {
            bias,
            ..self
        }
    }

    pub fn forward(&mut self, t: &Vec<nn::Node>) -> Vec<nn::Node> {
        assert!(t.len() == self.in_dim, "Input dimension mismatch: expected {}, got {}", self.in_dim, t.len());
        let mut linear_out = Vec::with_capacity(self.out_dim);
        for i in 0..self.out_dim {
            linear_out.push(&t[0] * &self.weights[i * self.in_dim + 0]);
            if self.in_dim > 0 {
                for j in 1..self.in_dim {
                    linear_out[i] = &linear_out[i] + &t[j] * &self.weights[i * self.in_dim + j];
                }
            }
            if let Some(b) = &self.bias {
                linear_out[i] = &linear_out[i] + &b[i];
            }
        }
        let mut out = Vec::with_capacity(self.out_dim);
        if let Some(act) = &self.activation {
            for i in 0..self.out_dim {
                out.push(act(&linear_out[i]));
            }
        }
        else {
            out = linear_out;
        }
        out
    }
}


// ========================== Graph ==========================

struct GraphOpItem<'a>{
    op: &'a Box<dyn OpNode>,
    inputs: Vec<&'a Node>,
    output: &'a Node,
}
impl<'a> GraphOpItem<'a> {
    pub fn from(output: &'a Node) -> Option<Self> {
        if output.is_leaf() {
            return None;
        }
        let inputs = output.from.as_ref().unwrap().inputs();
        let op: &Box<dyn OpNode> = output.from.as_ref().unwrap();
        Some(GraphOpItem {
            op, 
            inputs,
            output,
        })
    }
}

/// Graph is a collection of operations and nodes traced from a given node.  
/// For faster forward passes and gradient operations.
pub struct Graph<'a> {
    op_chain: Vec<GraphOpItem<'a>>, 
    pub nodes: Vec<&'a Node>,
}

impl<'a> Graph<'a> {
    /// Construct a graph from the given node (should be the output node of a computation).
    /// This will traverse the graph from the given node to the leaf nodes.
    pub fn from_trace(node: &'a Node) -> Option<Self> {
        let first_op = GraphOpItem::from(node);
        if first_op.is_none() {
            return None;
        }

        // Trace the graph from root to leaf, using breadth first search
        let mut op_chain : Vec<GraphOpItem<'a>> = Vec::new();
        let mut to_add = vec![first_op.unwrap()];
        while !to_add.is_empty() {
            let mut next_level: Vec<GraphOpItem<'a>> = Vec::new();
            for this_level_op in to_add.drain(..) {
                for this_level_inp in &this_level_op.inputs {
                    if let Some(next_level_item) = GraphOpItem::from(this_level_inp){
                        next_level.push(next_level_item);
                    }
                }
                op_chain.push(this_level_op);
            }
            to_add.append(&mut next_level);
        }
        // leaf -> root
        op_chain.reverse();

        // de-duplicate the op_chain
        // only keep the first occurrence of each op 
        // TODO: proof this is correct and necessary...
        let mut op_chain_final: Vec<GraphOpItem<'a>> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for op_item in op_chain {
            if seen.contains(&op_item.op.address()) {
                continue;
            }
            seen.insert(op_item.op.address());
            op_chain_final.push(op_item);
        }

        // collect nodes
        let mut nodes_hash: HashMap<usize, &'a Node> = HashMap::new();
        for op_item in &op_chain_final {
            nodes_hash.insert(op_item.output.address(), op_item.output);
            for input in &op_item.inputs {
                nodes_hash.insert(input.address(), input);
            }
        }
        let mut nodes: Vec<&'a Node> = Vec::new();
        for (_, node) in nodes_hash.iter() {
            nodes.push(*node);
        }

        Some(Graph{
            op_chain: op_chain_final,
            nodes,
        })
    }

    /// Fast-forward the graph, from leaf to root, 
    /// without reallocating the nodes.
    pub fn forward(&mut self) {
        for op_item in &mut self.op_chain {
            op_item.output.get_unsafe_mut().value = op_item.op.forward_value();
        }
    }

    pub fn zero_grad(&mut self) {
        for n in &self.nodes {
            n.get_unsafe_mut().grad = 0.0;
        }
    }

    /// Scale the gradients of the nodes in the graph to (grad * factor).
    pub fn scale_grad(&mut self, factor: fp_t) {
        for n in &self.nodes {
            n.get_unsafe_mut().grad *= factor;
        }
    }

    /// Apply the gradients to the nodes in the graph by 
    /// set the value to (value + factor*grad)
    pub fn apply_grad(&mut self, factor: fp_t) {
        for n in &self.nodes {
            if n.requires_grad {
                n.get_unsafe_mut().value += factor * n.grad;
            }
        }
    }

    pub fn clip_grad(&mut self, min: fp_t, max: fp_t) {
        for n in &self.nodes {
            if n.requires_grad {
                n.get_unsafe_mut().grad = n.grad.clamp(min, max);
            }
        }
    }

    pub fn to_graphviz(&self) -> String {
        let mut t = String::new();
        t += "digraph G {\n";
        t += "  node [ shape=box, fixedsize=false, color=black, fontcolor=black, fontsize=12, fillcolor=white, style=filled ];\n";
        t += "  edge [ color=black ];\n";
        t += "  rankdir=TB;\n";
        t += "  nodesep=0.5;\n";

        let opnode_id = |op: &Box<dyn OpNode>| -> String {
            format!("{}", op.address())
        };

        let node_id = |node: &Node| -> String {
            format!("{}", node.address())
        };

        let draw_op_node = |op: &Box<dyn OpNode>| -> String {
            format!(
                "  {} [label=\"{}\", color=blue];\n", 
                opnode_id(op), 
                op.name()
                    .rsplit_once("Op")
                    .expect("OpNode name should start with Op")
                    .1
                )
        };

        let draw_node = |node: &Node| -> String {
            let format_val = |val: fp_t| -> String {
                if val.abs() < 1e-3 {
                    format!("{:.3e}", val)
                } else if val.abs() > 1e3 {
                    format!("{:.3e}", val)
                } else {
                    format!("{:.2}", val)
                }
            };
            let get_node_label = |node: &Node| -> String {
                let mut ret = String::new();
                // ret += &format!("{}@", node_id(node));
                ret += &format_val(node.value);
                if node.requires_grad && node.grad != 0.0 {
                    ret += &format!(", ∂={}", format_val(node.grad));
                }
                if !node.requires_grad {
                    ret += ", const";
                }
                ret
            };
            format!("  {} [label=\"{}\"];\n", node_id(node), get_node_label(node))
        };

        let mut all_nodes : HashMap<String, &Node> = HashMap::new();
        self.op_chain.iter()
            .map(|op_item| {
                all_nodes.insert(node_id(op_item.output), op_item.output);
                for n in &op_item.inputs {
                    all_nodes.insert(node_id(n), n);
                }
            }).count();
        
        for (_, node) in all_nodes.iter() {
            t += &draw_node(node);
        }

        self.op_chain.iter()
            .map(|op_item| {
                t += &draw_op_node(op_item.op);
            }).count();

        self.op_chain.iter()
            .map(|op_item| {
                let op_id = opnode_id(op_item.op);
                let output_id = node_id(op_item.output);
                t += &format!("  {} -> {};\n", op_id, output_id);
                for input in &op_item.inputs {
                    let input_id = node_id(input);
                    t += &format!("  {} -> {};\n", input_id, op_id);
                }
            }).count();
        t += "}\n";
        t
    }
}

/// Main module most users will interact with.
pub mod nn {
    use super::*;

    // bring the types into the nn module
    pub use super::Node;
    pub use super::RawNode;
    pub use super::Graph;
    pub use super::Linear;
    pub use super::fp_t;
    pub use super::functional;
    pub use super::{OpNode, Nodish, forward_op};    // for register new ops
    pub use super::{add, sub, mul, div, neg, pow, abs, log, ln, sin, cos, tan};

    /// Creates a new variable node with the given value.
    /// Gradients will be computed for this node during backpropagation.
    /// 
    /// # Examples
    /// ```
    /// use mgrad::nn;
    /// 
    /// let a = nn::variable(5);
    /// let b = nn::variable(3);
    /// let c = &a * &b;
    /// c.backward(1);
    /// 
    /// assert_eq!(c.value, 15.0);
    /// assert_eq!(a.grad, 3.0);
    /// ```
    pub fn variable<T: Number>(value: T) -> Node {
        let n = RawNode::new(value.to_fp());
        Node::from(n)
    }

    /// Creates a new constant node with the given value.
    /// Gradients will not be computed for this node during backpropagation.
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

    #[test]
    fn test_graph(){
        let a = nn::variable(1);
        let b = nn::variable(0);

        let c: Node = 2*a;
        let mut d: Node = b+1;
        let y = c.pow(&d);

        let y_val = y.value;

        let mut g = Graph::from_trace(&y).unwrap();
        d.set_value(0);     // alter middle node
        g.forward();

        assert_eq!(y.value, y_val);
    }

    #[test]
    fn test_complex1(){

        let x = nn::variable(3);
        let y = nn::variable(4);
        let n4 = &x * &x;
        let n5 = n4 * &y;
        let n6 = &y +2;
        let n7: Node = n5 + n6;
        n7.backward(1.);

        assert_close(x.grad, 24., 1e-3);
        assert_close(y.grad, 10., 1e-3);
    }

    #[test]
    fn test_complex2(){

        let a = nn::variable(-4);
        let b = nn::variable(2);

        let c = &a + &b;
        let d = &a * &b + &b.pow(3);

        let c = &c + (&c + 1);
        let c = &c + 1 + c + (-&a);

        let d = &d + &d * 2 + nn::functional::relu(&(&b + &a));
        let d = &d + 3 * &d + nn::functional::relu(&(&b -&a));

        let e: nn::Node = c - d;
        let f = e.pow(2);

        let g = &f / 2;
        let g1: nn::Node = &g + 10 / f;

        g1.backward(1);

        assert_close(a.grad, 138.8338, 1e-3);
        assert_close(b.grad, 645.5773, 1e-3);

    }

    #[test]
    fn test_linear_layer() {
        let mut layer = Linear::new(2, 3).with_bias().with_activation(nn::functional::sigmoid);
        let input = vec![nn::variable(1.0), nn::variable(2.0)];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_linear_reg() {
        let mut layer = Linear::new(2, 1).with_bias().with_activation(nn::functional::sigmoid);
        let input = vec![nn::constant(1.0), nn::constant(2.0)];
        let output = layer.forward(&input).get(0).unwrap().shadow();

        for i in &mut layer.weights.iter() {
            i.get_unsafe_mut().value=1.;
        }
        if let Some(b) = &mut layer.bias {
            for i in b.iter() {
                i.get_unsafe_mut().value=1.;
            }
        }

        let aim = nn::constant(1.0);
        let loss = (&output - &aim).pow(2);
        let n_iter = 1000;
        let lr = 1e-3;
        let mut graph = nn::Graph::from_trace(&loss).unwrap();

        for _ in 0..n_iter {
            graph.forward();
            loss.backward(1.0);
            graph.apply_grad(-1.0 * lr);
            graph.zero_grad();
        }

        println!("output: {}", output.value);
        println!("aim: {}", aim.value);
        assert!((output.value - aim.value).abs() < 5e-2);
    }


}
