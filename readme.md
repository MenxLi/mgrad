
A minimal automatic differentiation library, just for fun...  
In 300 lines of code (`src/nn.h` and `src/*.cc`), no dependencies. 

Two demos are provided: 
- `demo.cc`: compute the gradient of a function and export computational graph.
- `demo_mlp.cc`: train a neural network for classification.

Use the following command to compile and run the demos:
```sh
g++ -std=c++17 -O3 src/*.cc demo[_mlp].cc
./a.out
```

Below shows the `demo_mlp.cc` training in action:  
![](https://limengxun-imagebed.oss-cn-wuhan-lr.aliyuncs.com/pic/mgrad_sample.gif)
