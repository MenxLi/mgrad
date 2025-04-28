/// Author: Li, Mengxun <mengxunli@whu.edu.cn>
/// Repository: https://github.com/menxli/mgrad
/// File created: 2025-04-26

use std::ops::{Deref, Add, Div, Mul, Sub, Neg};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[allow(non_camel_case_types)]
type fp_t = f32;

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
    fn forward_value(&self) -> fp_t; 

    /// calculate the value of the output node, 
    /// and return the output node, this does not set the `from` field of the output node
    fn forward(&self) -> Node {
        let n = RawNode::new(self.forward_value());
        Node::from(n)
    }

    /// should invoke backward on the input nodes with proper gradients
    fn backward(&self, grad: fp_t); 

    fn address(&self) -> usize {
        self as *const Self as *const () as usize
    }
}

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
impl Deref for Node {
    type Target = Rc<RawNode>;
    fn deref(&self) -> &Self::Target {
        &self.0
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
        impl $op<Node> for &$t {
            type Output = Node;
            fn $name(self, other: Node) -> Self::Output {
                OpImpl::$name_impl(&nn::constant(*self), &other)
            }
        }
        impl $op<&Node> for &$t {
            type Output = Node;
            fn $name(self, other: &Node) -> Self::Output {
                OpImpl::$name_impl(&nn::constant(*self), other)
            }
        }
        impl $op<$t> for Node {
            type Output = Node;
            fn $name(self, other: $t) -> Self::Output {
                OpImpl::$name_impl(&self, &nn::constant(other))
            }
        }
        impl $op<&$t> for Node {
            type Output = Node;
            fn $name(self, other: &$t) -> Self::Output {
                OpImpl::$name_impl(&self, &nn::constant(*other))
            }
        }
        impl $op<$t> for &Node {
            type Output = Node;
            fn $name(self, other: $t) -> Self::Output {
                OpImpl::$name_impl(self, &nn::constant(other))
            }
        }
        impl $op<&$t> for &Node {
            type Output = Node;
            fn $name(self, other: &$t) -> Self::Output {
                OpImpl::$name_impl(self, &nn::constant(*other))
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
    pub fn pow(&self, other: impl Nodish) -> Node {
        let op = OpPow {
            a: Node::clone(&self),
            b: other.to_node(),
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

    pub fn log(&self, base: impl Nodish) -> Node {
        let op = OpLog {
            base: base.to_node(),
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
pub struct Graph<'a> {
    op_chain: Vec<GraphOpItem<'a>>
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

        Some(Graph{
            op_chain: op_chain_final,
        })
    }

    /// Fast-forward the graph, from leaf to root, 
    /// without reallocating the nodes.
    pub fn forward(&mut self) {
        for op_item in &mut self.op_chain {
            op_item.output.get_unsafe_mut().value = op_item.op.forward_value();
        }
    }

    pub fn to_graphvis(&self) -> String {
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
}
