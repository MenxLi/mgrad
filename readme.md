
A minimal automatic differentiation library.  
Only 300 lines of code, no dependencies. Just for fun...

Two demos are provided, 
`demo.cc` and `demo_vector.cc`.  
The former is fitting a quadratic function, 
the latter is using MLP for binary classification. 

Use the following command to compile and run the demos:
```sh
g++ -std=c++11 -O3 src/*.cc demo.cc
./a.out
```

The computation graph will be saved in `model.mermaid`.