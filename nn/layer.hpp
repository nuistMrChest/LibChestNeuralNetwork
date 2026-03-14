#ifndef LAYER_HPP
#define LAYER_HPP

#include"matrix.hpp"
#include<functional>
#include<random>

namespace LibCN{
    template<Element T>struct Layer{
        std::function<Matrix<T>(const Matrix<T>&)>activation;
        std::function<Matrix<T>(const Matrix<T>&)>activation_d;
        size_t in_size;
        size_t out_size;
        Matrix<T>W;
        Matrix<T>b;
        Matrix<T>last_input;
        Matrix<T>z;
        bool sm;

        Layer(){
            in_size=0;
            out_size=0;
            W=Matrix<T>();
            b=Matrix<T>();
            last_input=Matrix<T>();
            z=Matrix<T>();
            sm=false;
        }

        Layer(size_t i,size_t o){
            in_size=i;
            out_size=o;
            W.resize(o,i);
            b.resize(o,1);
            last_input.resize(i,1);
            z.resize(o,1);
            sm=false;
        }

        Matrix<T>forward(const Matrix<T>&input){
            Matrix<T>res(out_size,1);
            last_input=input;
            if(input.h==in_size&&input.l==1)z=((W*input)+b);
            res=activation(z);
            return res;
        }

        Matrix<T>backward(const Matrix<T>&dl_da,const T&step){
            Matrix<T>res;
            Matrix<T>dl_dz=dl_da.hadamard(activation_d(z));
            res=W.transpose()*dl_dz;
            W-=step*(dl_dz*last_input.transpose());
            b-=step*dl_dz;
            return res;
        }

        Matrix<T>backward_dz(const Matrix<T>&dl_dz,const T&step){
            Matrix<T>res=W.transpose()*dl_dz;
            W-=step*(dl_dz*last_input.transpose());
            b-=step*dl_dz;
            return res;
        }
    
        void init(T low=T(-1),T high=T(1)){
            static std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<T>dist(low,high);
            for(size_t i=0;i<out_size;++i){
                for(size_t j=0;j<in_size;++j){
                    W(i,j)=dist(rng);
                }
                b(i,0)=dist(rng);
            }
        }
    };
}

#endif