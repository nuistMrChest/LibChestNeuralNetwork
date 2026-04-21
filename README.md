# LibCN 5.0

A lightweight, header-only neural network library written in modern C++.

LibCN is a small native C++ library for learning, experimentation, and simple CPU-side neural-network projects.  
The current **5.0** codebase provides:

- a 2D matrix container: `Matrix<T>`
- a 3D tensor container: `Tensor3d<T>`
- a 4D tensor alias for convolution kernels: `Tensor4d<T>`
- fully connected layers and networks: `MLPLayer<T>`, `MLP<T>`
- convolution layers and a simple CNN wrapper: `CNNLayer<T>`, `CNN<T>`
- built-in activation and loss functions
- parameter save/load at layer level
- optional multi-threaded acceleration for some heavy matrix and convolution workloads

LibCN focuses on readability, directness, and hackability rather than framework-scale abstraction.

---

## Features

- Header-only
- Modern C++ templates
- No external dependencies
- Dense matrix operations
- 3D tensor operations
- Flatten / deflatten helpers
- Single-kernel and multi-kernel convolution
- MLP support
- CNN layer support
- A simple `CNN + MLP` pipeline
- Common activations and losses
- Optional `softmax + cross_entropy` special path
- Optional multi-threading controlled by a compile-time macro

---

## Requirements

- **C++20** or newer
- A modern compiler such as:
  - GCC
  - Clang
  - MSVC

Example compilation on Linux:

```bash
g++ -std=c++20 -pthread example.cpp -o example
```

Because LibCN uses `std::thread` in several code paths, `-pthread` is recommended on GCC/Clang toolchains.

---

## Installation

LibCN is header-only.

Copy the files into your project with this structure:

```text
lib_chest_nn.hpp
nn/
    matrix.hpp
    layer.hpp
    activations.hpp
    network.hpp
    losses.hpp
    tensor_3d.hpp
```

Then include:

```cpp
#include "lib_chest_nn.hpp"
```

No separate build step or linking step is required beyond compiling your own source file.

---

## Current Core Types

### `Matrix<T>`

A 2D dense matrix type used by the MLP side of the library.

Main capabilities:

- element access with `operator()(i, j)`
- addition / subtraction
- scalar multiplication
- matrix multiplication with `operator*`
- transpose
- Hadamard product
- formatted printing

### `Tensor3d<T>`

A 3D tensor type used by the CNN side of the library.

Main capabilities:

- element access with `operator()(c, h, w)`
- addition / subtraction
- scalar multiplication
- Hadamard product
- flatten to `Matrix<T>`
- deflatten from `Matrix<T>`
- convolution with one kernel (`Tensor3d<T>`)
- convolution with multiple kernels (`Tensor4d<T>`)

### `Tensor4d<T>`

An alias:

```cpp
template<Element T>
using Tensor4d = std::vector<Tensor3d<T>>;
```

In practice, this is used as a set of convolution kernels, where each output channel stores one `Tensor3d<T>` kernel.

---

## Threading Model

LibCN 5.0 does **not** expose runtime thread control through something like `setThreadNum(...)`.

Instead, threading is controlled by the macro `thread_num` in `matrix.hpp`:

```cpp
#ifndef thread_num
#define thread_num 10
#endif
```

That means:

- by default, the library may use up to `10` worker threads in supported heavy operations
- you can override this before including the library:

```cpp
#define thread_num 4
#include "lib_chest_nn.hpp"
```

or disable those threaded paths:

```cpp
#define thread_num 0
#include "lib_chest_nn.hpp"
```

### Where threading is used

The current code may use multiple threads in:

- large matrix multiplication in `Matrix<T>::operator*(const Matrix<T>&)`
- multi-kernel convolution in `Tensor3d<T>::convolution(const Tensor4d<T>&, ...)`
- parts of `CNNLayer<T>::backward(...)`

### Threshold behavior

The library only uses the threaded path for sufficiently large workloads. Small workloads stay on the single-thread path to avoid thread-launch overhead.

---

## Activation Functions

The library provides matrix activations in `LibCN::Activations`:

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

It also provides 3D tensor variants for CNN usage:

- `relu_t`
- `relu_d_t`
- `leaky_relu_t`
- `leaky_relu_d_t`
- `sigmoid_t`
- `sigmoid_d_t`
- `tanh_t`
- `tanh_d_t`
- `identity_t`
- `identity_d_t`

---

## Loss Functions

Available in `LibCN::Losses`:

- `MSE`
- `MSE_d`
- `MAE`
- `MAE_d`
- `cross_entropy`
- `cross_entropy_d`

These losses currently operate on `Matrix<T>` values.

---

## MLP Overview

The fully connected side of the library is built from:

- `MLPLayer<T>`
- `MLP<T>`

Typical workflow:

1. create an `MLP<T>`
2. define each layer with `setLayer(...)`
3. assign activation functions with `setLayerFun(...)`
4. assign a loss with `setLoss(...)`
5. initialize parameters with `init(...)`
6. train with `train(...)`
7. run inference with `use(...)`

### Special `softmax + cross_entropy` path

LibCN includes a manual special case for the common output combination:

- last layer activation is `softmax`
- loss is `cross_entropy`

To enable it, set:

```cpp
mlp.layers.back().sm = true;
mlp.ce = true;
```

When both flags are set, the last layer backpropagates with:

```cpp
output - expected
```

instead of calling the ordinary loss derivative path.

---

## CNN Overview

The convolution side is built from:

- `CNNLayer<T>`
- `CNN<T>`

`CNNLayer<T>` handles:

- multi-channel convolution
- per-output-channel bias
- activation
- backward propagation for convolution kernels and input gradients

`CNN<T>` is a light wrapper that combines:

- a list of convolution layers in `cnn.layers`
- a public `MLP<T> mlp` classifier at the end

Training is done with:

```cpp
cnn.train(input_tensor, expected_output);
```

### Inference note

The current `CNN<T>` wrapper does **not** provide a `use(...)` helper.

For inference, the usual pattern is:

1. forward through each `CNNLayer`
2. flatten the final `Tensor3d`
3. call `cnn.mlp.use(...)`

---

## Quick Example

A minimal MLP example:

```cpp
#include <iostream>
#include "lib_chest_nn.hpp"

int main() {
    using T = double;

    LibCN::MLP<T> net(2, 2, 2, 0.1);

    net.setLayer(0, 2, 4);
    net.setLayer(1, 4, 2);

    net.setLayerFun(0, LibCN::Activations::tanh<T>, LibCN::Activations::tanh_d<T>);
    net.setLayerFun(1, LibCN::Activations::softmax<T>, LibCN::Activations::softmax_d<T>);

    net.setLoss(
        LibCN::Losses::cross_entropy<T>,
        LibCN::Losses::cross_entropy_d<T>
    );

    net.layers[1].sm = true;
    net.ce = true;

    net.init(-1.0, 1.0);

    LibCN::Matrix<T> x{{0.0}, {1.0}};
    LibCN::Matrix<T> y{{0.0}, {1.0}};
    LibCN::Matrix<T> first_layer_grad;

    for (int i = 0; i < 1000; ++i) {
        net.train(x, y, first_layer_grad);
    }

    std::cout << net.use(x) << "\n";
}
```

For a broader demonstration covering matrix operations, tensors, convolution, activations, losses, MLP, CNN layers, and a small CNN training example, see `example.cpp`.

---

## Project Structure

```text
lib_chest_nn.hpp
nn/
    matrix.hpp
    layer.hpp
    activations.hpp
    network.hpp
    losses.hpp
    tensor_3d.hpp
```

### File Overview

**lib_chest_nn.hpp**  
Main entry header.

**nn/matrix.hpp**  
2D matrix container, arithmetic, transpose, Hadamard product, and matrix multiplication.

**nn/tensor_3d.hpp**  
3D tensor container, flatten / deflatten, convolution, tensor aliases, and threshold helper.

**nn/activations.hpp**  
Built-in activation functions for `Matrix<T>` and `Tensor3d<T>`.

**nn/losses.hpp**  
Built-in loss functions for `Matrix<T>`.

**nn/layer.hpp**  
Definitions of `MLPLayer<T>` and `CNNLayer<T>`.

**nn/network.hpp**  
Definitions of `MLP<T>` and `CNN<T>`.

---

## Scope

LibCN 5.0 is currently suitable for:

- learning neural-network internals
- educational demos
- simple native C++ experiments
- small CPU-side projects
- custom experimentation with MLPs and small CNNs

It is **not** intended to compete with large production frameworks such as PyTorch or TensorFlow.

---

## Notes

- The API is intentionally low-level and transparent.
- Several implementation details are currently exposed as public members.
- Thread control is compile-time macro based, not runtime configurable.

---

## Documentation

- `README.md`: high-level overview and usage notes
- `api.md`: source-level API reference for LibCN 5.0

