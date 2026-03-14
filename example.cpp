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