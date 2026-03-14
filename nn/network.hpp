#ifndef NETWORK_HPP
#define NETWORK_HPP

#include"matrix.hpp"
#include"layer.hpp"
#include<vector>
#include<functional>
#include"activations.hpp"
#include"losses.hpp"
#include<iostream>

namespace LibCN{
    template<Element T>struct Network{
        size_t in_size;
        size_t out_size;
        std::vector<Layer<T>>layers;
        T step;
        std::function<T(const Matrix<T>&,const Matrix<T>&)>loss;
        std::function<Matrix<T>(const Matrix<T>&,const Matrix<T>&)>loss_d;
        bool ce;

        void train(const Matrix<T>&input,const Matrix<T>&expected){
            Matrix<T>last_output=input;
            Matrix<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            Matrix<T>last_grad;
            if(layers.back().sm&&ce){
                Matrix<T>dl_dz = output - expected;
                last_grad = layers.back().backward_dz(dl_dz, step);
                for(size_t i=0;i<layers.size()-1;i++){
                    size_t j=layers.size()-2-i;
                    last_grad = layers[j].backward(last_grad, step);
                }
            }
            else{
                Matrix<T>last_dl_da = loss_d(output, expected);
                for(size_t i=0;i<layers.size();i++){
                    size_t j=layers.size()-1-i;
                    last_dl_da = layers[j].backward(last_dl_da, step);
                }
            }
        }

        void train_p(const Matrix<T>&input,const Matrix<T>&expected){
            Matrix<T>last_output=input;
            Matrix<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            std::cout<<"Loss: "<<loss(output,expected)<<std::endl;
            Matrix<T>last_grad;
            if(ce&&layers.back().sm){
                Matrix<T>dl_dz = output-expected;
                last_grad = layers.back().backward_dz(dl_dz, step);
                for(size_t i=0;i<layers.size()-1;i++){
                    size_t j=layers.size()-2-i;
                    last_grad = layers[j].backward(last_grad, step);
                }
            }
            else{
                Matrix<T>last_dl_da = loss_d(output, expected);
                for(size_t i=0;i<layers.size();i++){
                    size_t j=layers.size()-1-i;
                    last_dl_da = layers[j].backward(last_dl_da, step);
                }
            }
        }

        void setLayer(size_t index,size_t i,size_t o){
            layers[index]=Layer<T>(i,o);
        }

        void setLayerFun(size_t index,const std::function<Matrix<T>(const Matrix<T>&)>&a,const std::function<Matrix<T>(const Matrix<T>&)>&a_d){
            layers[index].activation=a;
            layers[index].activation_d=a_d;
        }

        void setLoss(const std::function<T(const Matrix<T>&,const Matrix<T>&)>l,const std::function<Matrix<T>(const Matrix<T>&,const Matrix<T>&)>l_d){
            loss=l;
            loss_d=l_d;
        }

        Matrix<T>use(const Matrix<T>&input){
            Matrix<T>res;
            Matrix<T>last_output=input;
            Matrix<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            res=output;
            return res;
        }

        Network(){
            in_size=0;
            out_size=0;
            layers.resize(0);
            step=T{};
            ce=false;
        }

        Network(size_t layer_size,size_t in_size,size_t out_size,const T&step){
            this->in_size=in_size;
            this->out_size=out_size;
            this->step=step;
            this->layers.resize(layer_size);
            ce=false;
        }
    
        void init(T low=T(-1),T high=T(1)){
            for(size_t i=0;i<layers.size();i++)layers[i].init(low,high);
        }
    }; 
}

#endif