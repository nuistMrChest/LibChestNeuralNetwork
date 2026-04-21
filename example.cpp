#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "lib_chest_nn.hpp"

using std::cout;
using std::string;

namespace {
using T = double;

void section(const string& title) {
    cout << "\n========== " << title << " ==========" << "\n";
}

// Build a one-hot column vector with size n.
LibCN::Matrix<T> one_hot(std::size_t index, std::size_t n) {
    LibCN::Matrix<T> y(n, 1);
    for (std::size_t i = 0; i < n; ++i) y(i, 0) = 0.0;
    y(index, 0) = 1.0;
    return y;
}

// Manually run the CNN forward path for inference.
LibCN::Matrix<T> run_cnn_use(LibCN::CNN<T>& cnn, const LibCN::Tensor3d<T>& input) {
    LibCN::Tensor3d<T> x = input;
    for (std::size_t i = 0; i < cnn.layers.size(); ++i) {
        x = cnn.layers[i].forward(x);
    }
    return cnn.mlp.use(x.flatten());
}

// Demonstrate basic matrix operations.
void demo_matrix() {
    section("1. Basic matrix operations");

    LibCN::Matrix<T> A{{1.0, 2.0}, {3.0, 4.0}};
    LibCN::Matrix<T> B{{5.0, 6.0}, {7.0, 8.0}};

    cout << "A =\n" << A << "\n";
    cout << "B =\n" << B << "\n";
    cout << "A + B =\n" << (A + B) << "\n";
    cout << "A - B =\n" << (A - B) << "\n";
    cout << "A * 2 =\n" << (A * 2.0) << "\n";
    cout << "2 * A =\n" << (2.0 * A) << "\n";
    cout << "A^T =\n" << A.transpose() << "\n";
    cout << "A * B =\n" << (A * B) << "\n";
    cout << "A hadamard B =\n" << A.hadamard(B) << "\n";
}

// Demonstrate tensor operations, flatten/deflatten, and convolution.
void demo_tensor() {
    section("2. Tensor3d / Tensor4d / flatten / convolution");

    LibCN::Tensor3d<T> x{{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    }};

    LibCN::Tensor3d<T> y{{
        {9.0, 8.0, 7.0},
        {6.0, 5.0, 4.0},
        {3.0, 2.0, 1.0}
    }};

    cout << "x =\n" << x << "\n";
    cout << "y =\n" << y << "\n";
    cout << "x + y =\n" << (x + y) << "\n";
    cout << "x - y =\n" << (x - y) << "\n";
    cout << "x * 0.5 =\n" << (x * 0.5) << "\n";
    cout << "x hadamard y =\n" << x.hadamard(y) << "\n";

    LibCN::Matrix<T> flat = x.flatten();
    cout << "flatten(x) =\n" << flat << "\n";
    cout << "deflatten(flat, 1, 3, 3) =\n"
         << LibCN::Tensor3d<T>::deflatten(flat, 1, 3, 3) << "\n";

    LibCN::Tensor3d<T> kernel{{
        {1.0, 0.0},
        {0.0, -1.0}
    }};
    cout << "single-kernel convolution =\n" << x.convolution(kernel, 1, 0) << "\n";

    LibCN::Tensor4d<T> kernels;
    kernels.push_back(LibCN::Tensor3d<T>{{
        {1.0, 0.0},
        {0.0, -1.0}
    }});
    kernels.push_back(LibCN::Tensor3d<T>{{
        {0.0, 1.0},
        {-1.0, 0.0}
    }});

    cout << "multi-kernel convolution =\n" << x.convolution(kernels, 1, 0) << "\n";
}

// Demonstrate built-in activation functions and loss functions.
void demo_activations_and_losses() {
    section("3. Activations and losses");

    LibCN::Matrix<T> m{{-2.0}, {-0.5}, {0.5}, {2.0}};
    cout << "input =\n" << m << "\n";
    cout << "relu =\n" << LibCN::Activations::relu(m) << "\n";
    cout << "relu_d =\n" << LibCN::Activations::relu_d(m) << "\n";
    cout << "leaky_relu =\n" << LibCN::Activations::leaky_relu(m) << "\n";
    cout << "leaky_relu_d =\n" << LibCN::Activations::leaky_relu_d(m) << "\n";
    cout << "sigmoid =\n" << LibCN::Activations::sigmoid(m) << "\n";
    cout << "sigmoid_d =\n" << LibCN::Activations::sigmoid_d(m) << "\n";
    cout << "tanh =\n" << LibCN::Activations::tanh(m) << "\n";
    cout << "tanh_d =\n" << LibCN::Activations::tanh_d(m) << "\n";
    cout << "identity =\n" << LibCN::Activations::identity(m) << "\n";
    cout << "identity_d =\n" << LibCN::Activations::identity_d(m) << "\n";

    LibCN::Matrix<T> logits{{1.0}, {2.0}, {0.5}};
    LibCN::Matrix<T> prob = LibCN::Activations::softmax(logits);
    LibCN::Matrix<T> expected = one_hot(1, 3);
    cout << "softmax(logits) =\n" << prob << "\n";
    cout << "softmax_d(logits) =\n" << LibCN::Activations::softmax_d(logits) << "\n";
    cout << "MSE(prob, expected) = " << LibCN::Losses::MSE(prob, expected) << "\n";
    cout << "MAE(prob, expected) = " << LibCN::Losses::MAE(prob, expected) << "\n";
    cout << "cross_entropy(prob, expected) = "
         << LibCN::Losses::cross_entropy(prob, expected) << "\n";
    cout << "MSE_d =\n" << LibCN::Losses::MSE_d(prob, expected) << "\n";
    cout << "MAE_d =\n" << LibCN::Losses::MAE_d(prob, expected) << "\n";
    cout << "cross_entropy_d =\n" << LibCN::Losses::cross_entropy_d(prob, expected) << "\n";
}

// Demonstrate a single MLP layer, including parameter save/load.
void demo_mlp_layer() {
    section("4. MLPLayer forward / backward / save / load");

    LibCN::MLPLayer<T> layer(3, 2);
    layer.activation = LibCN::Activations::sigmoid<T>;
    layer.activation_d = LibCN::Activations::sigmoid_d<T>;
    layer.init(-0.5, 0.5);

    LibCN::Matrix<T> input{{0.2}, {0.7}, {1.0}};
    LibCN::Matrix<T> out = layer.forward(input);
    cout << "forward output =\n" << out << "\n";

    LibCN::Matrix<T> fake_grad{{0.3}, {-0.2}};
    LibCN::Matrix<T> prev_grad = layer.backward(fake_grad, 0.05);
    cout << "backward propagated gradient =\n" << prev_grad << "\n";

    LibCN::Matrix<T> saved_w = layer.saveWeight();
    LibCN::Matrix<T> saved_b = layer.saveBias();
    bool ok_w = layer.loadWeight(saved_w);
    bool ok_b = layer.loadBias(saved_b);
    cout << "loadWeight success = " << ok_w << "\n";
    cout << "loadBias success = " << ok_b << "\n";
}

// Train a small MLP on XOR and demonstrate the softmax + cross-entropy setup.
void demo_mlp_network() {
    section("5. MLP network: XOR training + softmax/cross-entropy special case");

    LibCN::MLP<T> mlp(2, 2, 2, 0.15);
    mlp.setLayer(0, 2, 4);
    mlp.setLayer(1, 4, 2);
    mlp.setLayerFun(0, LibCN::Activations::tanh<T>, LibCN::Activations::tanh_d<T>);
    mlp.setLayerFun(1, LibCN::Activations::softmax<T>, LibCN::Activations::softmax_d<T>);
    mlp.setLoss(LibCN::Losses::cross_entropy<T>, LibCN::Losses::cross_entropy_d<T>);
    mlp.layers[1].sm = true;
    mlp.ce = true;
    mlp.init(-1.0, 1.0);

    std::vector<LibCN::Matrix<T>> xs = {
        LibCN::Matrix<T>{{0.0}, {0.0}},
        LibCN::Matrix<T>{{0.0}, {1.0}},
        LibCN::Matrix<T>{{1.0}, {0.0}},
        LibCN::Matrix<T>{{1.0}, {1.0}}
    };
    std::vector<LibCN::Matrix<T>> ys = {
        one_hot(0, 2),
        one_hot(1, 2),
        one_hot(1, 2),
        one_hot(0, 2)
    };

    for (int epoch = 1; epoch <= 1500; ++epoch) {
        T loss_sum = 0.0;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            LibCN::Matrix<T> first_layer_grad;
            loss_sum += mlp.train(xs[i], ys[i], first_layer_grad);
        }
        if (epoch % 300 == 0) {
            cout << "epoch " << epoch << ", avg_loss = " << (loss_sum / xs.size()) << "\n";
        }
    }

    for (std::size_t i = 0; i < xs.size(); ++i) {
        cout << "XOR input " << i << " ->\n" << mlp.use(xs[i]) << "\n";
    }
}

// Demonstrate a single convolution layer, including backward propagation and save/load.
void demo_cnn_layer() {
    section("6. CNNLayer forward / backward / save / load");

    LibCN::Tensor3d<T> input{{
        {1.0, 0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0, 1.0},
        {1.0, 0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0, 1.0}
    }};

    LibCN::CNNLayer<T> conv(1, 4, 4, 2, 3, 3, 1, 0);
    conv.activation = LibCN::Activations::relu_t<T>;
    conv.activation_d = LibCN::Activations::relu_d_t<T>;
    conv.init(2, 1, 2, 2, -0.5, 0.5);

    LibCN::Tensor3d<T> out = conv.forward(input);
    cout << "conv output =\n" << out << "\n";

    LibCN::Tensor3d<T> fake_grad(2, 3, 3);
    for (std::size_t i = 0; i < fake_grad.v.size(); ++i) fake_grad.v[i] = 1.0;
    LibCN::Tensor3d<T> prev_grad = conv.backward(fake_grad, 0.03);
    cout << "conv backward propagated gradient =\n" << prev_grad << "\n";

    LibCN::Tensor4d<T> saved_k = conv.saveKernel();
    std::vector<T> saved_b = conv.saveBias();
    cout << "loadKernel success = " << conv.loadKernel(saved_k) << "\n";
    cout << "loadBias success = " << conv.loadBias(saved_b) << "\n";
}

// Train a tiny CNN made of one conv layer plus an MLP classifier.
void demo_cnn_network() {
    section("7. CNN network: conv + flatten + mlp");

    LibCN::Tensor3d<T> vertical{{
        {0.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 0.0}
    }};

    LibCN::Tensor3d<T> horizontal{{
        {0.0, 0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.0}
    }};

    std::vector<LibCN::Tensor3d<T>> xs{vertical, horizontal};
    std::vector<LibCN::Matrix<T>> ys{one_hot(0, 2), one_hot(1, 2)};

    LibCN::CNN<T> cnn(1, 1, 4, 4, 2, 3, 3, 0.08);
    cnn.layers[0] = LibCN::CNNLayer<T>(1, 4, 4, 2, 3, 3, 1, 0);
    cnn.layers[0].activation = LibCN::Activations::leaky_relu_t<T>;
    cnn.layers[0].activation_d = LibCN::Activations::leaky_relu_d_t<T>;
    cnn.layers[0].init(2, 1, 2, 2, -0.8, 0.8);

    cnn.mlp = LibCN::MLP<T>(2, 18, 2, 0.08);
    cnn.mlp.setLayer(0, 18, 6);
    cnn.mlp.setLayer(1, 6, 2);
    cnn.mlp.setLayerFun(0, LibCN::Activations::tanh<T>, LibCN::Activations::tanh_d<T>);
    cnn.mlp.setLayerFun(1, LibCN::Activations::softmax<T>, LibCN::Activations::softmax_d<T>);
    cnn.mlp.setLoss(LibCN::Losses::cross_entropy<T>, LibCN::Losses::cross_entropy_d<T>);
    cnn.mlp.layers[1].sm = true;
    cnn.mlp.ce = true;
    cnn.mlp.init(-0.8, 0.8);

    for (int epoch = 1; epoch <= 400; ++epoch) {
        T loss_sum = 0.0;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            loss_sum += cnn.train(xs[i], ys[i]);
        }
        if (epoch % 100 == 0) {
            cout << "epoch " << epoch << ", avg_loss = " << (loss_sum / xs.size()) << "\n";
        }
    }

    cout << "vertical ->\n" << run_cnn_use(cnn, vertical) << "\n";
    cout << "horizontal ->\n" << run_cnn_use(cnn, horizontal) << "\n";
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    demo_matrix();
    demo_tensor();
    demo_activations_and_losses();
    demo_mlp_layer();
    demo_mlp_network();
    demo_cnn_layer();
    demo_cnn_network();

    std::cout << "\nAll demos finished.\n";
    return 0;
}
