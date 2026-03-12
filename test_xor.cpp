#include"lib_chest_nn.hpp"
#include <iostream>

using namespace std;
using namespace LibCN;

int main()
{
    // XOR 网络结构
    // 2 -> 4 -> 1
    Network<float> net(2,2,1,0.001f);

    net.setLayer(0,2,4);
    net.setLayer(1,4,1);

    net.init();
    // 激活函数
    net.setLayFun(0,Activations::relu<float>,Activations::relu_d<float>);
    net.setLayFun(1,Activations::identity<float>,Activations::identity_d<float>);

    // XOR 数据
    Matrix<float> x1{{0},{0}};
    Matrix<float> x2{{0},{1}};
    Matrix<float> x3{{1},{0}};
    Matrix<float> x4{{1},{1}};

    Matrix<float> y1{{0}};
    Matrix<float> y2{{1}};
    Matrix<float> y3{{1}};
    Matrix<float> y4{{0}};

    cout<<"before training"<<endl;

    cout<<"0 xor 0 -> "<<net.use(x1)<<endl;
    cout<<"0 xor 1 -> "<<net.use(x2)<<endl;
    cout<<"1 xor 0 -> "<<net.use(x3)<<endl;
    cout<<"1 xor 1 -> "<<net.use(x4)<<endl;

    // 训练
    for(int i=0;i<100000;i++)
    {
        net.train(x1,y1);
        net.train(x2,y2);
        net.train(x3,y3);
        net.train(x4,y4);
    }

    cout<<"\nafter training"<<endl;

    cout<<"0 xor 0 -> "<<net.use(x1)<<endl;
    cout<<"0 xor 1 -> "<<net.use(x2)<<endl;
    cout<<"1 xor 0 -> "<<net.use(x3)<<endl;
    cout<<"1 xor 1 -> "<<net.use(x4)<<endl;

    return 0;
}