#ifndef LOSSES_HPP
#define LOSSES_HPP

#include"matrix.hpp"
#include<cmath>

namespace LibChestNN{
	namespace Losses{
		template<Element T>T MSE(const Matrix<T>&x,const Matrix<T>&e){
			T res=T();
			if(x.h==e.h&&x.l==e.l){
				T sum=T(0);
				for(size_t i=0;i<x.h;i++)
					for(size_t j=0;j<x.l;j++)
						sum+=(x(i,j)-e(i,j))*(x(i,j)-e(i,j));
				res=sum/T(2);
			}
			return res;
		}

		template<Element T>Matrix<T>MSE_d(const Matrix<T>&x,const Matrix<T>&e){
			return x-e;
		}

		template<Element T>T MAE(const Matrix<T>&x,const Matrix<T>&e){
			T res=T();
			if(x.h==e.h&&x.l==e.l){
				T sum=T(0);
				for(size_t i=0;i<x.h;i++)for(size_t j=0;j<x.l;j++){
					T tmp=(x(i,j)-e(i,j));
					if(tmp>T(0))sum+=tmp;
					else sum-=tmp;
				}
				res=sum/(x.h*x.l);
			}
			return res;
		}

		template<Element T>Matrix<T>MAE_d(const Matrix<T>&x,const Matrix<T>&e){
			Matrix<T>res;
			if(x.h==e.h&&x.l==e.l){
				res.resize(x.h,x.l);
				T scale=T(1)/T(x.h*x.l);
				for(size_t i=0;i<x.h;i++){
					for(size_t j=0;j<x.l;j++){
						if(x(i,j)>e(i,j))res(i,j)=scale;
						else if(x(i,j)<e(i,j))res(i,j)=T(-1)*scale;
						else res(i,j)=T(0);
					}
				}
			}
			return res;
		}

		template<Element T>T cross_entropy(const Matrix<T>&x,const Matrix<T>&e){
			T res=T();
			if(x.h==e.h&&x.l==e.l){
				constexpr T eps = static_cast<T>(1e-12);
				for(size_t i=0;i<x.h;++i){
					for(size_t j=0;j<x.l;++j){
						T v=x(i,j);
						if(v<eps)v=eps;
						if(v>static_cast<T>(1)-eps)v=static_cast<T>(1)-eps;
						res-=e(i,j)*std::log(v);
					}
				}
			}
			return res;
		}

		template<Element T>Matrix<T>cross_entropy_d(const Matrix<T>&x,const Matrix<T>&e){
			Matrix<T> res(x.h, x.l);
			if(x.h!=e.h||x.l!=e.l){
				return res;
			}
			constexpr T eps=static_cast<T>(1e-12);
			for(size_t i=0;i<x.h;++i){
				for(size_t j=0;j<x.l;++j){
					T v=x(i,j);
					if(v<eps)v=eps;
					if(v>static_cast<T>(1)-eps)v=static_cast<T>(1)-eps;
					res(i,j)=T(-1)*e(i,j)/v;
				}
			}
			return res;
		}
	}
}

#endif
