#ifndef LAYER_HPP
#define LAYER_HPP

#include"matrix.hpp"
#include<functional>

namespace LibCN{
    template<Element T>struct Layer{
        std::function<T(T)>activation;
        std::function<T(T)>activation_d;
        size_t in_size;
        size_t out_size;
        Matrix<T>W;
        Matrix<T>b;
        Matrix<T>last_input;
        Matrix<T>z;

        Layer(){
            in_size=0;
            out_size=0;
            W=Matrix<T>();
            b=Matrix<T>();
            last_input=Matrix<T>();
            z=Matrix<T>();
        }

        Layer(size_t i,size_t o){
            in_size=i;
            out_size=o;
            W.resize(o,i);
            b.resize(o,1);
            last_input.resize(i,1);
            z.resize(o,1);
        }

        Matrix<T>forward(const Matrix<T>&input){
            Matrix<T>res(out_size,1);
            last_input=input;
            if(input.h==in_size&&input.l==1)z=((W*input)+b);
            res=z.apply(activation);
            return res;
        }

        Matrix<T>backward(const Matrix<T>&dl_da,const T&step){
            Matrix<T>res;
            Matrix<T>dl_dz=dl_da.hadamard(z.apply(activation_d));
            res=W.transpose()*dl_dz;
            W-=step*(dl_dz*last_input.transpose());
            b-=step*dl_dz;
            return res;
        }
    };
}

#endif