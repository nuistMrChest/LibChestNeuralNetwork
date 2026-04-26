#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include"matrix.hpp"
#include"tensor_3d.hpp"
#include<cmath>

namespace LibChestNN{
	namespace Activations{
		template<Element T>Matrix<T>relu(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=a(i,j)>T(0)?a(i,j):T(0);
			return res;
		}

		template<Element T>Tensor3d<T>relu_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=a(i,j,k)>T(0)?a(i,j,k):T(0);
			return res;
		}

		template<Element T>Matrix<T>relu_d(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=a(i,j)>T(0)?T(1):T(0);
			return res;
		}

		template<Element T>Tensor3d<T>relu_d_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=a(i,j,k)>T(0)?T(1):T(0);
			return res;
		}

		template<Element T>Matrix<T>leaky_relu(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=a(i,j)>T(0)?a(i,j):T(0.01)*a(i,j);
			return res;
		}

		template<Element T>Tensor3d<T>leaky_relu_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=a(i,j,k)>T(0)?a(i,j,k):T(0.01)*a(i,j,k);
			return res;
		}

		template<Element T>Matrix<T>leaky_relu_d(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=a(i,j)>T(0)?T(1):T(0.01);
			return res;
		}

		template<Element T>Tensor3d<T>leaky_relu_d_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=a(i,j,k)>T(0)?T(1):T(0.01);
			return res;
		}

		template<Element T>Matrix<T>sigmoid(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=T(1)/(T(1)+std::exp(T(-1)*a(i,j)));
			return res;
		}

		template<Element T>Tensor3d<T>sigmoid_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=T(1)/(T(1)+std::exp(T(-1)*a(i,j,k)));
			return res;
		}

		template<Element T>Matrix<T>sigmoid_d(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			Matrix<T>s=sigmoid(a);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=s(i,j)*(T(1)-s(i,j));
			return res;
		}

		template<Element T>Tensor3d<T>sigmoid_d_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			Tensor3d<T>s=sigmoid_t(a);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=s(i,j,k)*(T(1)-s(i,j,k));
			return res;
		}

		template<Element T>Matrix<T>tanh(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=std::tanh(a(i,j));
			return res;
		}

		template<Element T>Tensor3d<T>tanh_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=std::tanh(a(i,j,k));
			return res;
		}

		template<Element T>Matrix<T>tanh_d(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			Matrix<T>t=tanh(a);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=T(1)-t(i,j)*t(i,j);
			return res;
		}

		template<Element T>Tensor3d<T>tanh_d_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			Tensor3d<T>t=tanh_t(a);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=T(1)-t(i,j,k)*t(i,j,k);
			return res;
		}

		template<Element T>Matrix<T>identity(const Matrix<T>&a){
			return a;
		}

		template<Element T>Tensor3d<T>identity_t(const Tensor3d<T>&a){
			return a;
		}

		template<Element T>Matrix<T>identity_d(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=T(1);
			return res;
		}

		template<Element T>Tensor3d<T>identity_d_t(const Tensor3d<T>&a){
			Tensor3d<T>res(a.c,a.h,a.l);
			for(size_t i=0;i<a.c;i++)
				for(size_t j=0;j<a.h;j++)
					for(size_t k=0;k<a.l;k++)
						res(i,j,k)=T(1);
			return res;
		}

		template<Element T>Matrix<T>softmax(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);

			T mx=a(0,0);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					if(a(i,j)>mx)mx=a(i,j);

			T sum=T(0);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++){
					res(i,j)=std::exp(a(i,j)-mx);
					sum+=res(i,j);
				}

			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)/=sum;

			return res;
		}

		template<Element T>Matrix<T>softmax_d(const Matrix<T>&a){
			Matrix<T>res(a.h,a.l);
			Matrix<T>s=softmax(a);
			for(size_t i=0;i<a.h;i++)
				for(size_t j=0;j<a.l;j++)
					res(i,j)=s(i,j)*(T(1)-s(i,j));
			return res;
		}
	}
}

#endif
