#include "./nn/network.hpp"
#include "./nn/activations.hpp"
#include <iostream>

using namespace std;
using namespace LibCN;

int main()
{
    // 创建网络：2层
    // 输入 2 → 隐藏 3 → 输出 1
    Network<float> net(2,2,1,0.1f);

    // 设置层结构
    net.setLayer(0,2,3);
    net.setLayer(1,3,1);

    // 设置激活函数
    net.setLayFun(0,Activations::relu<float>,Activations::relu_d<float>);
    net.setLayFun(1,Activations::identity<float>,Activations::identity_d<float>);

    // 输入
    Matrix<float> x{
        {0.5},
        {1.0}
    };

    // 期望输出
    Matrix<float> y{
        {1.0}
    };

    cout<<"initial output:"<<endl;
    cout<<net.use(x)<<endl;

    // 训练一次
    net.train(x,y);

    cout<<"after one train:"<<endl;
    cout<<net.use(x)<<endl;

    return 0;
}