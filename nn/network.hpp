#ifndef NETWORK_HPP
#define NETWORK_HPP

#include"matrix.hpp"
#include"layer.hpp"
#include<vector>
#include<functional>
#include"activations.hpp"
#include"losses.hpp"
#include<iostream>
#include"tensor_3d.hpp"

namespace LibChestNN{
	template<Element T>struct MLP{
		size_t in_size;
		size_t out_size;
		std::vector<MLPLayer<T>>layers;
		T step;
		std::function<T(const Matrix<T>&,const Matrix<T>&)>loss;
		std::function<Matrix<T>(const Matrix<T>&,const Matrix<T>&)>loss_d;
		bool ce;

		T train(
		const Matrix<T>&input,
		const Matrix<T>&expected,
		Matrix<T>&l_dl_da
		){
			T res;
			Matrix<T>last_output=input;
			Matrix<T>output;
			for(size_t i=0;i<layers.size();i++){
				output=layers[i].forward(last_output);
				last_output=output;
			}
			res=loss(output,expected);
			Matrix<T>last_grad;
			if(layers.back().sm&&ce){
				Matrix<T>dl_dz=output-expected;
				last_grad=layers.back().backward_dz(dl_dz,step);
				l_dl_da=last_grad;
				for(size_t i=0;i<layers.size()-1;i++){
					size_t j=layers.size()-2-i;
					last_grad=layers[j].backward(last_grad, step);
					if(i==layers.size()-2)l_dl_da=last_grad;
				}
			}
			else{
				Matrix<T>last_dl_da=loss_d(output, expected);
				for(size_t i=0;i<layers.size();i++){
					size_t j=layers.size()-1-i;
					last_dl_da=layers[j].backward(last_dl_da,step);
					if(i==layers.size()-1)l_dl_da=last_dl_da;
				}
			}
			return res;
		}

		void setLayer(size_t index,size_t i,size_t o){
			layers[index]=MLPLayer<T>(i,o);
		}

		void setLayerFun(
			size_t index,
			const std::function<Matrix<T>(const Matrix<T>&)>&a,
			const std::function<Matrix<T>(const Matrix<T>&)>&a_d
		){
			layers[index].activation=a;
			layers[index].activation_d=a_d;
		}

		void setLoss(
			const std::function<T(
				const Matrix<T>&,
				const Matrix<T>&
			)>l,
			const std::function<Matrix<T>(
				const Matrix<T>&,
				const Matrix<T>&
			)>l_d
		){
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

		MLP(){
			in_size=0;
			out_size=0;
			layers.resize(0);
			step=T{};
			ce=false;
		}

		MLP(
		size_t layer_size,
		size_t in_size,
		size_t out_size,
		const T&step
		){
			this->in_size=in_size;
			this->out_size=out_size;
			this->step=step;
			this->layers.resize(layer_size);
			ce=false;
		}
	
		void init(T low=T(-1),T high=T(1)){
			for(size_t i=0;i<layers.size();i++)
				layers[i].init(low,high);
		}
	}; 

	template<Element T>struct CNN{
		size_t i_c,i_h,i_l;
		size_t o_c,o_h,o_l;
		std::vector<CNNLayer<T>>layers;
		T step;

		MLP<T>mlp;

		T train(const Tensor3d<T>&input,const Matrix<T>&expected){
			T res;
			Tensor3d<T>last_input=input;
			Tensor3d<T>output;
			for(size_t i=0;i<layers.size();i++){
				output=layers[i].forward(last_input);
				last_input=output;
			}
			Matrix<T>m_l_dl_da;
			res=mlp.train(output.flatten(),expected,m_l_dl_da);
			Tensor3d<T>last_dl_da=Tensor3d<T>::deflatten(
				m_l_dl_da,
				layers[layers.size()-1].o_c,
				layers[layers.size()-1].o_h,
				layers[layers.size()-1].o_l
			);
			for(size_t i=0;i<layers.size();i++){
				size_t j=layers.size()-1-i;
				last_dl_da=layers[j].backward(last_dl_da,step);
			}
			return res;
		}

		CNN(){
			i_c=0;
			i_h=0;
			i_l=0;
			o_c=0;
			o_h=0;
			o_l=0;
			layers.resize(0);
			step=T();
		}

		CNN(
			size_t layer_size,
			size_t i_c,
			size_t i_h,
			size_t i_l,
			size_t o_c,
			size_t o_h,
			size_t o_l,
			const T&step
		){
			this->i_c=i_c;
			this->i_h=i_h;
			this->i_l=i_l;
			this->o_c=o_c;
			this->o_h=o_h;
			this->o_l=o_l;
			this->step=step;
			this->layers.resize(layer_size);
		}

		void init(T low=T(-1),T high=T(1)){
			for(size_t i=0;i<layers.size();i++)
				layers[i].init(low,high);
		}
	};
}

#endif
