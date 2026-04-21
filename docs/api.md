# LibCN 5.0 API Reference

This document describes the public interfaces present in the current **LibCN 5.0** source.

To use the whole library, include:

```cpp
#include "lib_chest_nn.hpp"
```

The entry header includes:

```cpp
./nn/matrix.hpp
./nn/layer.hpp
./nn/activations.hpp
./nn/network.hpp
./nn/losses.hpp
./nn/tensor_3d.hpp
```

---

## Namespace

All main symbols are in:

```cpp
namespace LibCN
```

Activation and loss functions are in:

```cpp
namespace LibCN::Activations
namespace LibCN::Losses
```

---

# Element Concept

Defined in `nn/matrix.hpp`.

```cpp
template<typename T>
concept Element = requires(T a, T b, std::iostream& os) {
    { a + b }  -> std::same_as<T>;
    { a += b } -> std::same_as<T&>;
    { a - b }  -> std::same_as<T>;
    { a -= b } -> std::same_as<T&>;
    { a * b }  -> std::same_as<T>;
    { a *= b } -> std::same_as<T&>;
    { os << a }-> std::same_as<std::ostream&>;
    { a > b }  -> std::same_as<bool>;
    { a < b }  -> std::same_as<bool>;
    { a >= b } -> std::same_as<bool>;
    { a <= b } -> std::same_as<bool>;
    { a == b } -> std::same_as<bool>;
    { a != b } -> std::same_as<bool>;
    { a / b }  -> std::same_as<T>;
    T{};
    T{0};
    T();
    T(0);
};
```

All container and network templates require `T` to satisfy `Element`.

---

# Thread Macro

Defined in `nn/matrix.hpp`.

```cpp
#ifndef thread_num
#define thread_num 10
#endif
```

This is a compile-time macro, not a runtime field.

It is used by:

- `Matrix<T>::operator*(const Matrix<T>&)`
- `Tensor3d<T>::convolution(const Tensor4d<T>&, size_t, size_t) const`
- `CNNLayer<T>::backward(const Tensor3d<T>&, const T&)`

To override it:

```cpp
#define thread_num 4
#include "lib_chest_nn.hpp"
```

To disable those threaded paths:

```cpp
#define thread_num 0
#include "lib_chest_nn.hpp"
```

---

# Matrix

Defined in:

```cpp
nn/matrix.hpp
```

## Overview

`Matrix<T>` is the 2D dense matrix type used throughout the MLP side of LibCN.

## Data Members

```cpp
std::vector<T> v;
size_t h, l;
```

Where:

- `h` = height / row count
- `l` = length / column count

## Constructors

### Default constructor

```cpp
Matrix()
```

Creates an empty matrix with `h = 0`, `l = 0`.

### Sized constructor

```cpp
Matrix(size_t h, size_t l)
```

Allocates a matrix of shape `h x l`.

### From nested vector

```cpp
Matrix(std::vector<std::vector<T>>& a)
```

Builds a matrix from nested `std::vector`s.

### Copy constructor

```cpp
Matrix(const Matrix<T>& a)
```

### Copy assignment

```cpp
Matrix<T>& operator=(const Matrix<T>& a)
```

### Initializer-list constructor

```cpp
Matrix(std::initializer_list<std::initializer_list<T>> init)
```

Example:

```cpp
Matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
```

## Element Access

```cpp
T& operator()(size_t i, size_t j)
const T& operator()(size_t i, size_t j) const
```

No bounds checking is performed.

## Output

```cpp
friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& a)
```

Prints the matrix in a readable multi-line format. An empty matrix prints as `{ NULL }`.

## Basic Operations

### Resize

```cpp
void resize(size_t h, size_t l)
```

### Transpose

```cpp
Matrix<T> transpose() const
```

### Addition

```cpp
Matrix<T> operator+(const Matrix<T>& a) const
Matrix<T>& operator+=(const Matrix<T>& a)
```

### Subtraction

```cpp
Matrix<T> operator-(const Matrix<T>& a) const
Matrix<T>& operator-=(const Matrix<T>& a)
```

### Scalar multiplication

```cpp
Matrix<T> operator*(const T& a) const
Matrix<T>& operator*=(const T& a)
friend Matrix<T> operator*(const T& a, const Matrix<T>& b)
```

### Matrix multiplication

```cpp
Matrix<T> operator*(const Matrix<T>& a) const
Matrix<T>& operator*=(const Matrix<T>& a)
```

If `this->l != a.h`, multiplication returns an empty matrix.

The implementation may use multiple threads for large matrices, depending on `thread_num`.

### Hadamard product

```cpp
Matrix<T> hadamard(const Matrix<T>& a) const
```

Element-wise multiplication for same-shaped matrices.

## Helper used by threaded multiplication

```cpp
static void subMatrixMultiplication(
    size_t f,
    size_t m,
    size_t n,
    size_t p,
    Matrix<T>& res,
    const Matrix<T>& a,
    const Matrix<T>& b
)
```

This is public because it is inside the struct, but it is mainly an implementation helper for threaded matrix multiplication.

## Alias

```cpp
template<typename T>
using Tensor2d = Matrix<T>;
```

---

# over_threshold

Defined in:

```cpp
nn/tensor_3d.hpp
```

## Signature

```cpp
bool over_threshold(size_t threshold, std::initializer_list<size_t> xs)
```

## Purpose

Checks whether the product of a list of dimensions reaches or exceeds `threshold`, with overflow-aware early exit logic.

This helper is used by the convolution and CNN backward code to decide whether the threaded path should be taken.

---

# Tensor3d and Tensor4d

Defined in:

```cpp
nn/tensor_3d.hpp
```

## Aliases

```cpp
template<Element T>
using Tensor4d = std::vector<Tensor3d<T>>;
```

`Tensor4d<T>` is used as a list of convolution kernels.

---

## Tensor3d Overview

`Tensor3d<T>` is the main 3D tensor type used by the CNN side of LibCN.

## Data Members

```cpp
std::vector<T> v;
size_t c, h, l;
```

Where:

- `c` = channel count
- `h` = height
- `l` = width / length

## Constructors

### Default constructor

```cpp
Tensor3d()
```

### Sized constructor

```cpp
Tensor3d(size_t c, size_t h, size_t l)
```

### From matrix

```cpp
Tensor3d(const Matrix<T>& a)
```

Creates a 1-channel tensor from a matrix.

### Copy constructor

```cpp
Tensor3d(const Tensor3d<T>& a)
```

### Assignment from tensor

```cpp
Tensor3d<T>& operator=(const Tensor3d<T>& a)
```

### Assignment from matrix

```cpp
Tensor3d& operator=(const Matrix<T>& a)
```

### 3D initializer-list constructor

```cpp
Tensor3d(
    std::initializer_list<
        std::initializer_list<
            std::initializer_list<T>
        >
    > init
)
```

Example:

```cpp
Tensor3d<double> x{{
    {1.0, 2.0},
    {3.0, 4.0}
}};
```

This produces a tensor with `c = 1`.

## Element Access

```cpp
T& operator()(size_t i, size_t j, size_t k)
T operator()(size_t i, size_t j, size_t k) const
```

## Indexed access by pointer to coordinates

```cpp
T& visit(size_t* cord)
T visit(size_t* cord) const
```

## Output

```cpp
friend std::ostream& operator<<(std::ostream& os, const Tensor3d<T>& a)
```

## Resize

```cpp
void resize(size_t c, size_t h, size_t l)
```

## Arithmetic

### Addition

```cpp
Tensor3d<T> operator+(const Tensor3d<T>& a) const
Tensor3d<T>& operator+=(const Tensor3d<T>& a)
```

### Subtraction

```cpp
Tensor3d<T> operator-(const Tensor3d<T>& a) const
Tensor3d<T>& operator-=(const Tensor3d<T>& a)
```

### Scalar multiplication

```cpp
Tensor3d<T> operator*(const T& a) const
friend Tensor3d<T> operator*(const T& a, const Tensor3d<T>& b)
Tensor3d<T>& operator*=(const T& a)
```

### Hadamard product

```cpp
Tensor3d<T> hadamard(const Tensor3d<T>& a) const
```

## Flatten and deflatten

### Flatten

```cpp
Matrix<T> flatten()
```

Returns a column vector of shape `(c * h * l) x 1`.

### Deflatten

```cpp
static Tensor3d<T> deflatten(const Matrix<T>& a, size_t c, size_t h, size_t l)
```

Rebuilds a tensor from a flattened matrix.

## Convolution with one kernel

```cpp
Matrix<T> convolution(const Tensor3d<T>& a, size_t stride, size_t padding)
```

Treats `a` as a single convolution kernel and returns one output feature map as a `Matrix<T>`.

## Convolution with many kernels

```cpp
Tensor3d<T> convolution(const Tensor4d<T>& a, size_t stride, size_t padding) const
```

Treats `a` as one kernel per output channel and returns a tensor of output feature maps.

The implementation may use multiple threads for large workloads.

## Threaded helper

```cpp
static void con_for(
    Tensor3d<T>& res,
    const Tensor4d<T>& a,
    size_t stride,
    size_t padding,
    const Tensor3d<T>& b,
    size_t from,
    size_t to
)
```

This is primarily an internal helper for threaded multi-kernel convolution.

---

# Activations

Defined in:

```cpp
nn/activations.hpp
```

All activation functions are in:

```cpp
LibCN::Activations
```

---

## Matrix activations

All matrix activation functions have the form:

```cpp
template<Element T>
Matrix<T> f(const Matrix<T>& a)
```

### Available functions

```cpp
relu
relu_d
leaky_relu
leaky_relu_d
sigmoid
sigmoid_d
tanh
tanh_d
identity
identity_d
softmax
softmax_d
```

### Notes

- `softmax` applies max-subtraction stabilization before exponentiation.
- `softmax_d` returns the element-wise form `s * (1 - s)`, not the full Jacobian matrix.

---

## Tensor3d activations

Tensor3d activation functions have the form:

```cpp
template<Element T>
Tensor3d<T> f(const Tensor3d<T>& a)
```

### Available functions

```cpp
relu_t
relu_d_t
leaky_relu_t
leaky_relu_d_t
sigmoid_t
sigmoid_d_t
tanh_t
tanh_d_t
identity_t
identity_d_t
```

---

# Losses

Defined in:

```cpp
nn/losses.hpp
```

All loss functions are in:

```cpp
LibCN::Losses
```

All current loss functions operate on `Matrix<T>`.

---

## MSE

```cpp
template<Element T>
T MSE(const Matrix<T>& x, const Matrix<T>& e)
```

Computes half squared error:

```cpp
sum((x - e)^2) / 2
```

for matching shapes.

## MSE derivative

```cpp
template<Element T>
Matrix<T> MSE_d(const Matrix<T>& x, const Matrix<T>& e)
```

Returns:

```cpp
x - e
```

## MAE

```cpp
template<Element T>
T MAE(const Matrix<T>& x, const Matrix<T>& e)
```

Computes mean absolute error for matching shapes.

## MAE derivative

```cpp
template<Element T>
Matrix<T> MAE_d(const Matrix<T>& x, const Matrix<T>& e)
```

Returns element-wise:

- `+1 / (h*l)` when `x(i,j) > e(i,j)`
- `-1 / (h*l)` when `x(i,j) < e(i,j)`
- `0` when equal

## Cross entropy

```cpp
template<Element T>
T cross_entropy(const Matrix<T>& x, const Matrix<T>& e)
```

Computes:

```cpp
- sum(e(i,j) * log(v))
```

with `v` clamped into `[1e-12, 1 - 1e-12]`.

## Cross entropy derivative

```cpp
template<Element T>
Matrix<T> cross_entropy_d(const Matrix<T>& x, const Matrix<T>& e)
```

Returns:

```cpp
- e(i,j) / v
```

with the same clamp rule.

---

# MLPLayer

Defined in:

```cpp
nn/layer.hpp
```

## Overview

`MLPLayer<T>` represents one fully connected layer.

## Data Members

```cpp
std::function<Matrix<T>(const Matrix<T>&)> activation;
std::function<Matrix<T>(const Matrix<T>&)> activation_d;
size_t in_size;
size_t out_size;
Matrix<T> W;
Matrix<T> b;
Matrix<T> last_input;
Matrix<T> z;
bool sm;
```

### Shapes

For `MLPLayer(i, o)`:

- `W` has shape `o x i`
- `b` has shape `o x 1`
- `last_input` has shape `i x 1`
- `z` has shape `o x 1`

### `sm`

`sm` defaults to `false`.

It is used as a flag for the `softmax + cross_entropy` special training path in `MLP<T>`.

## Constructors

### Default constructor

```cpp
MLPLayer()
```

### Sized constructor

```cpp
MLPLayer(size_t i, size_t o)
```

## Forward

```cpp
Matrix<T> forward(const Matrix<T>& input)
```

Performs:

```cpp
z = W * input + b
activation(z)
```

when the input shape is `in_size x 1`.

## Backward

```cpp
Matrix<T> backward(const Matrix<T>& dl_da, const T& step)
```

Computes the ordinary backpropagation path:

```cpp
dl_dz = dl_da.hadamard(activation_d(z))
res   = W.transpose() * dl_dz
W    -= step * (dl_dz * last_input.transpose())
b    -= step * dl_dz
```

Returns the gradient propagated to the previous layer.

## Backward with precomputed `dL/dz`

```cpp
Matrix<T> backward_dz(const Matrix<T>& dl_dz, const T& step)
```

Used for the `softmax + cross_entropy` special case.

## Initialization

```cpp
void init(T low = T(-1), T high = T(1))
```

Initializes `W` and `b` from a uniform distribution on `[low, high]`.

## Parameter save/load

```cpp
Matrix<T> saveWeight()
Matrix<T> saveBias()
bool loadWeight(const Matrix<T>& W)
bool loadBias(const Matrix<T>& b)
```

`loadWeight` and `loadBias` return `true` only when the input shape matches the stored parameter shape.

---

# CNNLayer

Defined in:

```cpp
nn/layer.hpp
```

## Overview

`CNNLayer<T>` represents one convolution layer.

## Data Members

```cpp
std::function<Tensor3d<T>(const Tensor3d<T>&)> activation;
std::function<Tensor3d<T>(const Tensor3d<T>&)> activation_d;
Tensor4d<T> kernel;
size_t i_c, i_h, i_l;
size_t o_c, o_h, o_l;
size_t stride, padding;
std::vector<T> b;
Tensor3d<T> z;
Tensor3d<T> last_input;
```

## Constructors

### Default constructor

```cpp
CNNLayer()
```

### Sized constructor

```cpp
CNNLayer(
    size_t i_c,
    size_t i_h,
    size_t i_l,
    size_t o_c,
    size_t o_h,
    size_t o_l,
    size_t s,
    size_t p
)
```

Where:

- `i_*` describe input shape
- `o_*` describe output shape
- `s` is stride
- `p` is padding

## Kernel initialization

```cpp
void init(
    size_t c_o,
    size_t c_i,
    size_t h,
    size_t l,
    T low = T(-1),
    T high = T(1)
)
```

Typical usage:

```cpp
conv.init(output_channels, input_channels, kernel_h, kernel_w, -0.5, 0.5);
```

## Forward

```cpp
Tensor3d<T> forward(const Tensor3d<T>& input)
```

Performs:

1. convolution with `kernel`
2. per-output-channel bias addition
3. activation

## Backward

```cpp
Tensor3d<T> backward(const Tensor3d<T>& dl_da, const T& step)
```

Computes:

- input gradient
- kernel gradient update
- bias gradient update

Returns the gradient with respect to the previous layer input.

The implementation may use multiple threads for large workloads.

## Thread helpers

```cpp
static void da_for(...)
static void grad_for(...)
```

These are public because they live inside the struct, but they are mainly internal helpers for threaded backward propagation.

## Parameter save/load

```cpp
Tensor4d<T> saveKernel()
std::vector<T> saveBias()
bool loadKernel(const Tensor4d<T>& K)
bool loadBias(const std::vector<T>& b)
```

These names intentionally match the current source spelling.

---

# MLP

Defined in:

```cpp
nn/network.hpp
```

## Overview

`MLP<T>` is the high-level fully connected network type.

## Data Members

```cpp
size_t in_size;
size_t out_size;
std::vector<MLPLayer<T>> layers;
T step;
std::function<T(const Matrix<T>&, const Matrix<T>&)> loss;
std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)> loss_d;
bool ce;
```

### `ce`

`ce` defaults to `false`.

It is used together with `layers.back().sm` to enable the `softmax + cross_entropy` special path.

## Constructors

### Default constructor

```cpp
MLP()
```

### Sized constructor

```cpp
MLP(size_t layer_size, size_t in_size, size_t out_size, const T& step)
```

## Configuration

### Set one layer

```cpp
void setLayer(size_t index, size_t i, size_t o)
```

### Set activation functions

```cpp
void setLayerFun(
    size_t index,
    const std::function<Matrix<T>(const Matrix<T>&)>& a,
    const std::function<Matrix<T>(const Matrix<T>&)>& a_d
)
```

### Set loss

```cpp
void setLoss(
    const std::function<T(const Matrix<T>&, const Matrix<T>&)> l,
    const std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)> l_d
)
```

### Initialize all layers

```cpp
void init(T low = T(-1), T high = T(1))
```

## Inference

```cpp
Matrix<T> use(const Matrix<T>& input)
```

Feeds `input` through all layers.

## Training

```cpp
T train(
    const Matrix<T>& input,
    const Matrix<T>& expected,
    Matrix<T>& l_dl_da
)
```

Performs one training step and returns the loss value.

The third argument is an output parameter that receives the gradient propagated to the input of the first MLP layer.

### Ordinary path

When the last layer is not marked as softmax-specialized:

```cpp
last_dl_da = loss_d(output, expected)
```

and each layer backpropagates through `backward(...)`.

### Special `softmax + cross_entropy` path

When both:

```cpp
layers.back().sm == true
ce == true
```

the last layer uses:

```cpp
dl_dz = output - expected
```

and calls `backward_dz(...)`.

---

# CNN

Defined in:

```cpp
nn/network.hpp
```

## Overview

`CNN<T>` is a lightweight wrapper that combines convolution layers with a public `MLP<T>` classifier.

## Data Members

```cpp
size_t i_c, i_h, i_l;
size_t o_c, o_h, o_l;
std::vector<CNNLayer<T>> layers;
T step;
MLP<T> mlp;
```

## Constructors

### Default constructor

```cpp
CNN()
```

### Sized constructor

```cpp
CNN(
    size_t layer_size,
    size_t i_c,
    size_t i_h,
    size_t i_l,
    size_t o_c,
    size_t o_h,
    size_t o_l,
    const T& step
)
```

## Training

```cpp
T train(const Tensor3d<T>& input, const Matrix<T>& expected)
```

Current workflow:

1. forward through every `CNNLayer`
2. flatten the last `Tensor3d`
3. train `mlp` on that flattened matrix
4. deflatten the MLP gradient back into tensor form
5. backpropagate through CNN layers

Returns the loss value from the inner `mlp.train(...)`.

## Initialization

```cpp
void init(T low = T(-1), T high = T(1))
```

Calls `init(...)` on every convolution layer in `layers`.

## Inference note

`CNN<T>` does **not** provide a `use(...)` helper in the current source.

The common inference pattern is:

```cpp
Tensor3d<T> x = input;
for (size_t i = 0; i < cnn.layers.size(); ++i) {
    x = cnn.layers[i].forward(x);
}
Matrix<T> y = cnn.mlp.use(x.flatten());
```

---

# Typical Usage Patterns

## MLP

```cpp
LibCN::MLP<double> net(2, 2, 2, 0.1);

net.setLayer(0, 2, 4);
net.setLayer(1, 4, 2);

net.setLayerFun(0, LibCN::Activations::tanh<double>, LibCN::Activations::tanh_d<double>);
net.setLayerFun(1, LibCN::Activations::softmax<double>, LibCN::Activations::softmax_d<double>);

net.setLoss(
    LibCN::Losses::cross_entropy<double>,
    LibCN::Losses::cross_entropy_d<double>
);

net.layers[1].sm = true;
net.ce = true;

net.init(-1.0, 1.0);
```

## CNN + MLP

```cpp
LibCN::CNN<double> cnn(1, 1, 4, 4, 2, 3, 3, 0.05);

cnn.layers[0] = LibCN::CNNLayer<double>(1, 4, 4, 2, 3, 3, 1, 0);
cnn.layers[0].activation = LibCN::Activations::relu_t<double>;
cnn.layers[0].activation_d = LibCN::Activations::relu_d_t<double>;
cnn.layers[0].init(2, 1, 2, 2, -0.5, 0.5);

cnn.mlp = LibCN::MLP<double>(1, 18, 2, 0.05);
cnn.mlp.setLayer(0, 18, 2);
cnn.mlp.setLayerFun(
    0,
    LibCN::Activations::softmax<double>,
    LibCN::Activations::softmax_d<double>
);
cnn.mlp.setLoss(
    LibCN::Losses::cross_entropy<double>,
    LibCN::Losses::cross_entropy_d<double>
);
cnn.mlp.layers[0].sm = true;
cnn.mlp.ce = true;
cnn.mlp.init(-0.5, 0.5);
```

---

# Source-Level Notes

These notes are included to keep the API reference honest to the current 5.0 headers.

- The library uses a compile-time macro named `thread_num`, which is simple but globally visible and may conflict with user code that uses the same macro name.
- `softmax_d` is an element-wise approximation, not the full softmax Jacobian.
- `CNN<T>` provides `train(...)` but not a top-level `use(...)` helper.

This document describes the current source as written, not an idealized future API.
