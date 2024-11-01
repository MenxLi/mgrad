
A minimal automatic differentiation library.  
Less than 400 lines of code, no dependencies. Just for fun...

Two demos are provided: 
- `demo.cc`: compute the gradient of a function and export computational graph.
- `demo_mlp.cc`: train a neural network for classification.

Use the following command to compile and run the demos:
```sh
g++ -std=c++17 -O3 src/*.cc demo.cc
./a.out
```

The computational graph will be saved in `model.mermaid`, 
you may visualize it using [Mermaid Live Editor](https://mermaid.live).