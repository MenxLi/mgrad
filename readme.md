
A minimal automatic differentiation library. Just for fun...

```cpp
nn::Graph g;

auto x = g.variable(1);
auto y = x.pow(2) * (x.sin() + 1);
y = y.log();

g.forward();
g.backward(y);

// y = log(x^2 * (sin(x) + 1))
// dy/dx should be ~ 2.29341
std::cout << x.grad() << std::endl;
```

Two demos are provided: 
- `demo.cc`: compute the gradient of a function and export computational graph.
- `demo_mlp.cc`: train a neural network for classification.

Use the following command to compile and run the demos:
```sh
g++ -std=c++17 -O3 src/*.cc demo[_mlp].cc
./a.out
```

<!-- Below shows the `demo_mlp.cc` training in action:  
![](https://limengxun-imagebed.oss-cn-wuhan-lr.aliyuncs.com/pic/mgrad_sample1.gif) -->