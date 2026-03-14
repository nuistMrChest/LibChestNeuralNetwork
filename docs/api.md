# LibCN API Documentation

This document describes the public interfaces of **Lib Chest NeuralNetwork (LibCN)**.

LibCN is a header-only C++ neural network library designed for educational purposes and experimentation.

To use the entire library, simply include:

```cpp
#include "lib_chest_nn.hpp"
```

All core headers are located inside the `nn/` directory but are automatically included by `lib_chest_nn.hpp`.

---

# Matrix

Defined in:

```cpp
nn/matrix.hpp
```

## Overview

`Matrix<T>` is the core mathematical container used throughout LibCN.

In v2.0.0, the matrix storage is implemented as:

```cpp
std::vector<T> v;
```

with explicit shape fields:

```cpp
size_t h;
size_t l;
```

Elements are mapped by coordinates rather than nested `operator[]` access.

---

## Template Requirements

The template type `T` must satisfy the `Element` concept.

It must support:

```cpp
+  +=
-  -=
*  *=
/
> < >= <= == !=
std::ostream << value
```

---

## Data Members

```cpp
std::vector<T> v;
size_t h, l;
```

- `v` stores matrix elements in contiguous memory
- `h` is the number of rows
- `l` is the number of columns

---

## Constructors

### Default constructor

```cpp
Matrix()
```

Creates an empty matrix.

Result:

```cpp
h = 0
l = 0
v.size() = 0
```

---

### Shape constructor

```cpp
Matrix(size_t h, size_t l)
```

Creates a matrix with shape `h × l` and resizes internal storage to `h * l`.

---

### From `std::vector<std::vector<T>>`

```cpp
Matrix(std::vector<std::vector<T>>& a)
```

Builds a matrix from a nested vector.

Behavior in the current implementation:

- `h = a.size()`
- `l = a[0].size()`
- values are copied row by row into the internal one-dimensional vector

This constructor takes a **non-const lvalue reference**.

---

### Copy constructor

```cpp
Matrix(const Matrix<T>& a)
```

Creates a deep copy of another matrix.

---

### Initializer-list constructor

```cpp
Matrix(std::initializer_list<std::initializer_list<T>> init)
```
```

Allows creation like:

```cpp
Matrix<float> a{{1,2,3},{4,5,6}};
```

Behavior in the current implementation:

- `h = init.size()`
- `l = init.begin()->size()`
- elements are appended row by row to `v`

The implementation assumes row sizes are consistent.

---

## Assignment

```cpp
Matrix<T>& operator=(const Matrix<T>& a)
```

Copies shape and storage.

---

## Element Access

### Mutable access

```cpp
T& operator()(size_t i, size_t j)
```

### Const access

```cpp
const T& operator()(size_t i, size_t j) const
```

Elements are accessed as:

```cpp
matrix(i, j)
```

Internally this maps to:

```cpp
v[i * l + j]
```

---

## Output

Matrices can be printed with `std::ostream`.

```cpp
friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& a)
```

If the matrix is empty, output is:

```cpp
{ NULL }
```

---

## resize

```cpp
void resize(size_t h, size_t l)
```

Changes matrix dimensions and resizes the internal storage to `h * l`.

---

## transpose

```cpp
Matrix<T> transpose() const
```

Returns the transposed matrix.

---

## Operators

### Addition

```cpp
Matrix<T> operator+(const Matrix<T>& a) const
Matrix<T>& operator+=(const Matrix& a)
```

Element-wise addition. If dimensions do not match, `operator+` returns an empty matrix and `operator+=` leaves the object unchanged.

---

### Subtraction

```cpp
Matrix<T> operator-(const Matrix<T>& a) const
Matrix<T>& operator-=(const Matrix& a)
```

Element-wise subtraction. If dimensions do not match, `operator-` returns an empty matrix and `operator-=` leaves the object unchanged.

---

### Scalar multiplication

```cpp
Matrix<T> operator*(const T& a) const
Matrix<T>& operator*=(const T& a)
friend Matrix<T> operator*(const T& a, const Matrix<T>& b)
```

Multiplies every element by a scalar.

---

### Matrix multiplication

```cpp
Matrix<T> operator*(const Matrix<T>& a) const
Matrix<T>& operator*=(const Matrix<T>& a)
```

Performs standard matrix multiplication.

If dimensions do not match, `operator*` returns an empty matrix.

---

## Hadamard Product

```cpp
Matrix<T> hadamard(const Matrix<T>& a) const
```

Performs element-wise multiplication.

If dimensions do not match, returns an empty matrix.

---

## Removed in v2.0.0

The following members from the old API are no longer present:

```cpp
append(...)
apply(...)
```

---

# Layer

Defined in:

```cpp
nn/layer.hpp
```

## Overview

`Layer<T>` represents a fully connected layer.

Each layer stores:

- activation function
- activation derivative function
- weight matrix `W`
- bias vector `b`
- cached input `last_input`
- cached pre-activation output `z`
- specialization flag `sm`

---

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

### `sm`

`sm` defaults to `false`.

It is used as a flag indicating that the layer is the softmax output layer in the specialized `softmax + cross entropy` training path.

---

## Constructors

### Default constructor

```cpp
Layer()
```

Creates an empty layer.

Initial state:

- `in_size = 0`
- `out_size = 0`
- all matrices are empty
- `sm = false`

---

### Sized constructor

```cpp
Layer(size_t i, size_t o)
```

Creates a layer with:

- `in_size = i`
- `out_size = o`
- `W` of shape `o × i`
- `b` of shape `o × 1`
- `last_input` of shape `i × 1`
- `z` of shape `o × 1`
- `sm = false`

---

## Activation Function Interface

In v2.0.0, activation functions use the following form:

```cpp
Matrix<T>(const Matrix<T>&)
```

This applies to both:

```cpp
activation
activation_d
```

This change allows activations such as `softmax`, which operate on a full matrix rather than purely element-wise scalar input.

---

## init

```cpp
void init(T low = T(-1), T high = T(1))
```

Initializes `W` and `b` with values sampled from a uniform distribution over `[low, high]`.

Implementation uses:

```cpp
static std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<T> dist(low, high);
```

---

## forward

```cpp
Matrix<T> forward(const Matrix<T>& input)
```

Computes:

```cpp
z = W * input + b
a = activation(z)
```

Behavior in the current implementation:

- `last_input = input`
- if `input.h == in_size && input.l == 1`, then `z` is updated
- returns `activation(z)`

---

## backward

```cpp
Matrix<T> backward(const Matrix<T>& dl_da, const T& step)
```

Standard backpropagation path.

Input:

- `dl_da`: derivative of loss with respect to the layer output activation
- `step`: learning rate

Computation:

```cpp
dl_dz = dl_da.hadamard(activation_d(z))
res   = W.transpose() * dl_dz
W    -= step * (dl_dz * last_input.transpose())
b    -= step * dl_dz
```

Returns gradient for the previous layer.

---

## backward_dz

```cpp
Matrix<T> backward_dz(const Matrix<T>& dl_dz, const T& step)
```

Special backpropagation path used when the caller already has `dL/dz` for the current layer.

Computation:

```cpp
res = W.transpose() * dl_dz
W  -= step * (dl_dz * last_input.transpose())
b  -= step * dl_dz
```

Returns gradient for the previous layer.

This function is used by the specialized `softmax + cross entropy` path in `Network<T>`.

---

# Activations

Defined in:

```cpp
nn/activations.hpp
```

All activation functions are inside:

```cpp
LibCN::Activations
```

All current activation functions use the signature:

```cpp
Matrix<T> f(const Matrix<T>& a)
```

---

## relu

```cpp
template<Element T> Matrix<T> relu(const Matrix<T>& a)
```

Applies:

```cpp
max(0, x)
```

for each element.

---

## relu_d

```cpp
template<Element T> Matrix<T> relu_d(const Matrix<T>& a)
```

Derivative of ReLU:

- `1` when `x > 0`
- `0` otherwise

---

## leaky_relu

```cpp
template<Element T> Matrix<T> leaky_relu(const Matrix<T>& a)
```

Uses slope `0.01` for negative values.

---

## leaky_relu_d

```cpp
template<Element T> Matrix<T> leaky_relu_d(const Matrix<T>& a)
```

Derivative of leaky ReLU:

- `1` when `x > 0`
- `0.01` otherwise

---

## sigmoid

```cpp
template<Element T> Matrix<T> sigmoid(const Matrix<T>& a)
```

Applies:

```cpp
1 / (1 + exp(-x))
```

---

## sigmoid_d

```cpp
template<Element T> Matrix<T> sigmoid_d(const Matrix<T>& a)
```

Computes derivative using:

```cpp
s = sigmoid(a)
s * (1 - s)
```

---

## tanh

```cpp
template<Element T> Matrix<T> tanh(const Matrix<T>& a)
```

Applies `std::tanh` element-wise.

---

## tanh_d

```cpp
template<Element T> Matrix<T> tanh_d(const Matrix<T>& a)
```

Computes derivative using:

```cpp
t = tanh(a)
1 - t * t
```

---

## identity

```cpp
template<Element T> Matrix<T> identity(const Matrix<T>& a)
```

Returns the input matrix directly.

---

## identity_d

```cpp
template<Element T> Matrix<T> identity_d(const Matrix<T>& a)
```

Returns a matrix of ones with the same shape.

---

## softmax

```cpp
template<Element T> Matrix<T> softmax(const Matrix<T>& a)
```

Current implementation:

1. finds the maximum element in `a`
2. computes `exp(a(i,j) - max)`
3. sums all exponentials
4. divides each element by the total sum

This is the numerically stabilized form using max-subtraction.

---

## softmax_d

```cpp
template<Element T> Matrix<T> softmax_d(const Matrix<T>& a)
```

Current implementation computes:

```cpp
s = softmax(a)
s * (1 - s)
```

for each element.

This is only an **approximate derivative form** and not the full Jacobian of softmax. Use it carefully.

---

# Losses

Defined in:

```cpp
nn/losses.hpp
```

All loss functions are inside:

```cpp
LibCN::Losses
```

---

## MSE

```cpp
template<Element T> T MSE(const Matrix<T>& x, const Matrix<T>& e)
```

If shapes match, computes:

```cpp
sum((x - e)^2) / 2
```

Otherwise returns default-constructed `T()`.

---

## MSE_d

```cpp
template<Element T> Matrix<T> MSE_d(const Matrix<T>& x, const Matrix<T>& e)
```

Computes:

```cpp
x - e
```

---

## MAE

```cpp
template<Element T> T MAE(const Matrix<T>& x, const Matrix<T>& e)
```

If shapes match, computes mean absolute error:

```cpp
sum(abs(x - e)) / (x.h * x.l)
```

Otherwise returns default-constructed `T()`.

---

## MAE_d

```cpp
template<Element T> Matrix<T> MAE_d(const Matrix<T>& x, const Matrix<T>& e)
```

If shapes match, returns element-wise:

- `1` when `x(i,j) >= e(i,j)`
- `-1` otherwise

If shapes do not match, returns an empty matrix.

---

## cross_entropy

```cpp
template<Element T> T cross_entropy(const Matrix<T>& x, const Matrix<T>& e)
```

If shapes match, computes:

```cpp
- sum(e(i,j) * log(v))
```

where `v` is `x(i,j)` clamped to:

```cpp
[1e-12, 1 - 1e-12]
```

Otherwise returns default-constructed `T()`.

---

## cross_entropy_d

```cpp
template<Element T> Matrix<T> cross_entropy_d(const Matrix<T>& x, const Matrix<T>& e)
```

If shapes match, computes element-wise:

```cpp
- e(i,j) / v
```

where `v` is `x(i,j)` clamped to:

```cpp
[1e-12, 1 - 1e-12]
```

If shapes do not match, returns a matrix of shape `x.h × x.l` because the current implementation constructs `res(x.h, x.l)` before checking shape equality.

---

# Network

Defined in:

```cpp
nn/network.hpp
```

## Overview

`Network<T>` represents a feed-forward neural network composed of multiple `Layer<T>` objects.

---

## Data Members

```cpp
size_t in_size;
size_t out_size;
std::vector<Layer<T>> layers;
T step;
std::function<T(const Matrix<T>&,const Matrix<T>&)> loss;
std::function<Matrix<T>(const Matrix<T>&,const Matrix<T>&)> loss_d;
bool ce;
```

### `ce`

`ce` defaults to `false`.

It is used as a flag for the specialized `softmax + cross entropy` training path.

When `ce == true` and `layers.back().sm == true`, training uses the specialized last-layer backpropagation path.

---

## Constructors

### Default constructor

```cpp
Network()
```

Creates an empty network.

Initial state:

- `in_size = 0`
- `out_size = 0`
- `layers` empty
- `step = T{}`
- `ce = false`

---

### Sized constructor

```cpp
Network(size_t layer_size, size_t in_size, size_t out_size, const T& step)
```

Initializes:

- network input size
- network output size
- learning rate
- layer container resized to `layer_size`
- `ce = false`

Layers are created as default-constructed layers and must be configured separately.

---

## setLayer

```cpp
void setLayer(size_t index, size_t i, size_t o)
```

Replaces `layers[index]` with:

```cpp
Layer<T>(i, o)
```

---

## setLayerFun

```cpp
void setLayerFun(
    size_t index,
    const std::function<Matrix<T>(const Matrix<T>&)>& a,
    const std::function<Matrix<T>(const Matrix<T>&)>& a_d)
```

Assigns activation function and derivative function for one layer.

---

## setLoss

```cpp
void setLoss(
    const std::function<T(const Matrix<T>&,const Matrix<T>&)> l,
    const std::function<Matrix<T>(const Matrix<T>&,const Matrix<T>&)> l_d)
```

Sets the loss function and its derivative.

---

## init

```cpp
void init(T low = T(-1), T high = T(1))
```

Calls `init(low, high)` on every layer.

---

## use

```cpp
Matrix<T> use(const Matrix<T>& input)
```

Runs a forward pass through all layers and returns the final output.

---

## train

```cpp
void train(const Matrix<T>& input, const Matrix<T>& expected)
```

Performs one training step.

### Forward phase

The input is passed through all layers sequentially.

### Backward phase

Two paths exist.

#### Ordinary path

Used when the following condition is false:

```cpp
layers.back().sm && ce
```

Then:

```cpp
last_dl_da = loss_d(output, expected)
```

and backpropagation proceeds by calling `backward(...)` from the last layer to the first.

#### Specialized softmax + cross entropy path

Used when:

```cpp
layers.back().sm && ce
```

Then the last layer receives:

```cpp
dl_dz = output - expected
```

and the last layer backpropagates through:

```cpp
layers.back().backward_dz(dl_dz, step)
```

All previous layers still use the ordinary `backward(...)` path.

---

## train_p

```cpp
void train_p(const Matrix<T>& input, const Matrix<T>& expected)
```

Same overall behavior as `train(...)`, but prints the current loss first:

```cpp
std::cout << "Loss: " << loss(output, expected) << std::endl;
```

It then performs backpropagation using the same two-path logic as `train(...)`.

---

# Usage Summary

To use the entire library:

```cpp
#include "lib_chest_nn.hpp"
```

Main components:

```cpp
nn/matrix.hpp
nn/layer.hpp
nn/activations.hpp
nn/losses.hpp
nn/network.hpp
```

No linking step is required.
