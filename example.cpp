#include "lib_chest_nn.hpp"
#include <iostream>

using namespace std;
using namespace LibCN;

int main()
{
    MLP<float> net(2, 2, 1, 0.05f);

    net.setLoss(Losses::MSE<float>, Losses::MSE_d<float>);

    net.setLayer(0, 2, 4);
    net.setLayer(1, 4, 1);

    net.init(-0.5f, 0.5f);

    net.setLayerFun(0, Activations::tanh<float>, Activations::tanh_d<float>);
    net.setLayerFun(1, Activations::sigmoid<float>, Activations::sigmoid_d<float>);

    Tensor<float> x1{{0,0}};
    Tensor<float> x2{{0,1}};
    Tensor<float> x3{{1,0}};
    Tensor<float> x4{{1,1}};

    cout<<x1.dimension<<endl;
    cout<<x2.dimension<<endl;
    cout<<x3.dimension<<endl;
    cout<<x4.dimension<<endl;

    Tensor<float> y1{0};
    Tensor<float> y2{1};
    Tensor<float> y3{1};
    Tensor<float> y4{0};

    cout<<y1.dimension<<endl;
    cout<<y2.dimension<<endl;
    cout<<y3.dimension<<endl;
    cout<<y4.dimension<<endl;

    cout<<y1.ascend().dimension<<endl;
    cout<<y2.ascend().dimension<<endl;
    cout<<y3.ascend().dimension<<endl;
    cout<<y4.ascend().dimension<<endl;

    cout << "before training" << endl;
    cout << "0 xor 0 -> " << net.use(x1) << endl;
    cout << "0 xor 1 -> " << net.use(x2) << endl;
    cout << "1 xor 0 -> " << net.use(x3) << endl;
    cout << "1 xor 1 -> " << net.use(x4) << endl;

    for(int i = 0; i < 50000; ++i)
    {
        if(i%2500==0)
        {
            net.train_p(x1, y1.ascend());
            net.train_p(x2, y2.ascend());
            net.train_p(x3, y3.ascend());
            net.train_p(x4, y4.ascend());
        }
        else
        {
            net.train(x1, y1.ascend());
            net.train(x2, y2.ascend());
            net.train(x3, y3.ascend());
            net.train(x4, y4.ascend());
        }
    }

    cout << "\nafter training" << endl;
    cout << "0 xor 0 -> " << net.use(x1) << endl;
    cout << "0 xor 1 -> " << net.use(x2) << endl;
    cout << "1 xor 0 -> " << net.use(x3) << endl;
    cout << "1 xor 1 -> " << net.use(x4) << endl;

    return 0;
}