#ifndef NETWORK_HPP
#define NETWORK_HPP

#include"matrix.hpp"
#include"layer.hpp"
#include<vector>
#include<functional>

namespace LibCN{
    template<Element T>struct Network{
        size_t in_size;
        size_t out_size;
        std::vector<Layer<T>>layers;
        T step;
        void train(const Matrix&a);
        void setLayer(size_t indez,size_t i,size_t o);
        void setLayFun(const std::function<T(T)>&a,const std::function<T(T)>&a_d);
        void use(const Matrix&a);

        Network();
        Network(size_t layer_num);
    };    
}

#endif