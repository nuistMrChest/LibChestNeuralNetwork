# Lib Chest NeuralNetwork (LibCN)

A lightweight header-only neural network library written in pure C++.

LibCN is a small C++ library built for learning, experimentation, and writing simple neural network code without heavy dependencies. It focuses on clarity and directness rather than framework-scale abstraction.

This project is mainly intended for educational use, small experiments, and as a reference implementation.

---

## Motivation

Large machine learning frameworks such as PyTorch and TensorFlow are powerful and highly optimized.

However, their C++ interfaces are often much heavier than what is needed for learning or small native projects.

LibCN tries to offer a simpler alternative:

- pure C++ implementation
- minimal abstraction
- easy to read and modify
- natural to use inside normal C++ code

This library is **not meant to replace industrial deep learning frameworks**. Its goal is to stay understandable.

---

## Goals

LibCN is designed to:

- help understand how neural networks work internally
- provide a simple matrix container for neural computation
- avoid external dependencies
- stay easy to integrate into existing C++ projects
- remain small enough to read through directly

---

## Features

- Header-only library
- Pure template implementation
- Requires only the C++ standard library
- No external dependencies
- No build system required
- Matrix operations
- Fully connected layers
- Multiple activation functions
- Multiple loss functions
- Optional specialized training path for Softmax output + Cross Entropy loss
- Optional loss printing during training

---

## New in v2.0.0

Compared with the previous version, LibCN v2.0.0 mainly introduces:

- `Matrix<T>` internal storage changed from nested vectors to a single `std::vector<T>` with coordinate mapping
- matrix element access changed to `matrix(i, j)`
- removed `append`
- removed `apply`
- activation function interface changed from scalar-in/scalar-out to matrix-in/matrix-out
- added `softmax`
- added selectable loss functions
- added `Network::setLoss(...)`
- added `Network::train_p(...)`
- added a specialized training path for **Softmax output layer + Cross Entropy loss**
- added `Layer::backward_dz(...)`
- added `Layer::sm` and `Network::ce`

---

## Design Philosophy

1. **Transparency over abstraction**  
   Code should be readable and understandable.

2. **Minimal dependencies**  
   Only the C++ standard library is used.

3. **Header-only simplicity**  
   No extra linking steps are required.

4. **C++ friendly interface**  
   The library is designed to feel natural in ordinary C++ code.

---

## Requirements

- C++20 or newer
- Any modern C++ compiler

Examples:

- GCC
- Clang
- MSVC

Example compilation:

```cpp
g++ -std=c++20 example.cpp -o example
```

---

## Installation

LibCN is header-only.

Simply copy the library into your project and include:

```cpp
#include "lib_chest_nn.hpp"
```

No installation script or package manager is required.

---

## Quick Example

A minimal XOR example:

```cpp

#include "lib_chest_nn.hpp"
#include <iostream>

using namespace std;
using namespace LibCN;

int main()
{
    Network<float> net(2, 2, 1, 0.05f);

    net.setLoss(Losses::MSE<float>, Losses::MSE_d<float>);

    net.setLayer(0, 2, 4);
    net.setLayer(1, 4, 1);

    net.init(-0.5f, 0.5f);

    net.setLayerFun(0, Activations::tanh<float>, Activations::tanh_d<float>);
    net.setLayerFun(1, Activations::sigmoid<float>, Activations::sigmoid_d<float>);

    Matrix<float> x1{{0},{0}};
    Matrix<float> x2{{0},{1}};
    Matrix<float> x3{{1},{0}};
    Matrix<float> x4{{1},{1}};

    Matrix<float> y1{{0}};
    Matrix<float> y2{{1}};
    Matrix<float> y3{{1}};
    Matrix<float> y4{{0}};

    cout << "before training" << endl;
    cout << "0 xor 0 -> " << net.use(x1) << endl;
    cout << "0 xor 1 -> " << net.use(x2) << endl;
    cout << "1 xor 0 -> " << net.use(x3) << endl;
    cout << "1 xor 1 -> " << net.use(x4) << endl;

    for(int i = 0; i < 50000; ++i)
    {
        if(i%2500==0)
        {
            net.train_p(x1, y1);
            net.train_p(x2, y2);
            net.train_p(x3, y3);
            net.train_p(x4, y4);
        }
        else
        {
            net.train(x1, y1);
            net.train(x2, y2);
            net.train(x3, y3);
            net.train(x4, y4);
        }
    }

    cout << "\nafter training" << endl;
    cout << "0 xor 0 -> " << net.use(x1) << endl;
    cout << "0 xor 1 -> " << net.use(x2) << endl;
    cout << "1 xor 0 -> " << net.use(x3) << endl;
    cout << "1 xor 1 -> " << net.use(x4) << endl;

    return 0;
}

```

---

## Softmax + Cross Entropy Specialized Path

LibCN contains a specialized training path for the common combination:

- output layer uses `softmax`
- loss uses cross entropy

This path is enabled by flags already present in the library:

```cpp
net.ce = true;
net.layers.back().sm = true;
```

When both conditions are true, `train(...)` and `train_p(...)` will use the specialized path.

In that path, the last layer receives:

```cpp
output - expected
```

as `dL/dz`, and the last layer backpropagation is performed through:

```cpp
backward_dz(...)
```

rather than the ordinary `backward(...)` path.

---

## Loss Functions

Current loss functions are provided in `LibCN::Losses`:

- `MSE`
- `MAE`
- `cross_entropy`

Their corresponding derivative functions are also provided.

Loss selection is done through:

```cpp
net.setLoss(loss_function, loss_derivative_function);
```

---

## Activation Functions

Current activation functions are provided in `LibCN::Activations`:

- `relu`
- `relu_d`
- `leaky_relu`
- `leaky_relu_d`
- `sigmoid`
- `sigmoid_d`
- `tanh`
- `tanh_d`
- `identity`
- `identity_d`
- `softmax`
- `softmax_d`

Note that `softmax_d` in the current implementation is an **approximate version**, not the full Jacobian form. Use it carefully.

---

## Project Structure

```text
lib_chest_nn.hpp
nn/
    matrix.hpp
    layer.hpp
    activations.hpp
    losses.hpp
    network.hpp
```

### File Overview

**lib_chest_nn.hpp**  
Main entry header.

**nn/matrix.hpp**  
Matrix type and matrix operations.

**nn/layer.hpp**  
Fully connected layer implementation.

**nn/activations.hpp**  
Activation functions and their derivatives.

**nn/losses.hpp**  
Loss functions and their derivatives.

**nn/network.hpp**  
High-level neural network structure.

---

## Current Status

LibCN is currently suitable for:

- learning neural networks
- educational demonstrations
- small experiments
- reference implementations

It is **not intended for production-scale deep learning workloads**.

---

## Author

MrChest / 石函

---

## License

This project is released under the MIT License.

See the `LICENSE` file for details.
