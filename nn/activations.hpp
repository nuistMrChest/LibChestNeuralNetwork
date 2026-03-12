#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include"matrix.hpp"
#include<cmath>

namespace LibCN{
    namespace Activations{
        template<Element T>T relu(T x){
            return x>T(0)?x:T(0);
        }

        template<Element T>T relu_d(T x){
            return x>T(0)?T(1):T(0);
        }

        template<Element T>T leaky_relu(T x,T alpha=T(0.01)){
            return x>T(0)?x:alpha*x;
        }

        template<Element T>T leaky_relu_d(T x,T alpha=T(0.01)){
            return x>T(0)?T(1):alpha;
        }

        template<Element T>T sigmoid(T x){
            return T(1)/(T(1)+std::exp(-x));
        }

        template<Element T>T sigmoid_d(T x){
            T s=sigmoid(x);
            return s*(T(1)-s);
        }

        template<Element T>T tanh(T x){
            return std::tanh(x);
        }

        template<Element T>T tanh_d(T x){
            T t=std::tanh(x);
            return T(1)-t*t;
        }

        template<Element T>T identity(T x){
            return x;
        }

        template<Element T>T identity_d(T x){
            return T(1);
        }
    }
}

#endif