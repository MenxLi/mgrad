
A minimal automatic differentiation library.  
Only 300 lines of code, no dependencies. Just for fun...

Run demo:
```sh
g++ -std=c++11 -O3 src/*.cc demo.cc
./a.out
```

This will run a small neural network fitting a quadratic function.  
The compute graph will be saved to `model.mermaid`.