A machine learning library written in C.

This was a personal project to write a pytorch-like library from scratch. All my own code using only the C standard library.

This is a work in progress. My goal is to add python bindings to perform training in Python in order to perform chess board evaluation using my game in this repo https://github.com/panu-hietanen/chess_engine.

Current key features:
- N-dimensional tensors (currently only 2d)
- Autograd for basic tensor operations
- Broadcasting
- Adam and SGD optimisers

*Written in C11 using Visual Studio 26 on Windows 10.*
