#ifndef TENSOR_3D_HPP
#define TENSOR_3D_HPP

#include<initializer_list>
#include<iostream>
#include<ostream>
#include<vector>
#include"matrix.hpp"
#include<thread>

namespace LibCN{
	bool over_threshold(size_t threshold,std::initializer_list<size_t>xs){
		size_t acc=1;
		for(size_t x:xs){
			if(x==0)return false;
			if(acc>threshold/x)return true;
			acc*=x;
		}
		return acc>=threshold;
	}

	template<Element T>struct Tensor3d;

	template<Element T>using Tensor4d=std::vector<Tensor3d<T>>;

	template<Element T>struct Tensor3d{
		std::vector<T>v;
		size_t c,h,l;

		Tensor3d(){
			c=0;
			h=0;
			l=0;
		}

		Tensor3d(size_t c,size_t h,size_t l){
			this->c=c;
			this->h=h;
			this->l=l;
			this->v.resize(c*h*l);
		}

		Tensor3d(const Matrix<T>&a){
			c=1;
			h=a.h;
			l=a.l;
			v=a.v;
		}

		Tensor3d(const Tensor3d<T>&a){
			this->c=a.c;
			this->h=a.h;
			this->l=a.l;
			this->v=a.v;
		}

		Tensor3d<T>&operator=(const Tensor3d<T>&a){
			this->c=a.c;
			this->h=a.h;
			this->l=a.l;
			this->v=a.v;
			return*this;
		}

		Tensor3d&operator=(const Matrix<T>&a){
			this->c=1;
			this->h=a.h;
			this->l=a.l;
			this->v=a.v;
			return*this;
		}

		Tensor3d(
			std::initializer_list<std::initializer_list<std::initializer_list<T>>>init
		){
			c=init.size();
			h=init.begin()->size();
			l=init.begin()->begin()->size();
			for(const auto&channel:init){
				for(const auto&row:channel){
					for(const auto&x:row){
						v.push_back(x);
					}
				}
			}
		}

		T&operator()(size_t i,size_t j,size_t k){
			return v[i*h*l+j*l+k];
		}

		T operator()(size_t i,size_t j,size_t k)const{
			return v[i*h*l+j*l+k];
		}

		T&visit(size_t*cord){
			return v[cord[0]*h*l+cord[1]*l+cord[2]];
		}

		T visit(size_t*cord)const{
			return v[cord[0]*h*l+cord[1]*l+cord[2]];
		}

		friend std::ostream&operator<<(std::ostream&os,const Tensor3d<T>&a){
			if(a.c==0){
				os<<"{ NULL }";
			}
			for(size_t i=0;i<a.c;i++){
				if(i==0)os<<"{\n";
				else os<<" ";
				for(size_t j=0;j<a.h;j++){
					if(j==0)os<<"{\n";
					else os<<"  ";
					for(size_t k=0;k<a.l;k++){
						os<<a(i,j,k)<<" ";
					}
					if(j==a.h-1)os<<"\n }";
					else os<<"\n";
				}
				if(i==a.c-1)os<<"\n}";
				else os<<"\n";
			}
			return os;
		}

		void resize(size_t c,size_t h,size_t l){
			this->c=c;
			this->h=h;
			this->l=l;
			v.resize(c*h*l);
		}

		Tensor3d<T>operator+(const Tensor3d<T>&a)const{
			Tensor3d<T>res;
			if(this->c==a.c&&this->h==a.h&&this->l==a.l){
				res.resize(c,h,l);
				for(size_t i=0;i<c;i++)
					for(size_t j=0;j<h;j++)
						for(size_t k=0;k<l;k++)
							res(i,j,k)=this->operator()(i,j,k)+a(i,j,k);
			}
			return res;
		}

		Tensor3d<T>&operator+=(const Tensor3d<T>&a){;
			if(this->c==a.c&&this->h==a.h&&this->l==a.l){
				for(size_t i=0;i<c;i++)
					for(size_t j=0;j<h;j++)
						for(size_t k=0;k<l;k++)
							this->operator()(i,j,k)+=a(i,j,k);
			}
			return*this;
		}

		Tensor3d<T>operator-(const Tensor3d<T>&a)const{
			Tensor3d<T>res;
			if(this->c==a.c&&this->h==a.h&&this->l==a.l){
				res.resize(c,h,l);
				for(size_t i=0;i<c;i++)
					for(size_t j=0;j<h;j++)
						for(size_t k=0;k<l;k++)
							res(i,j,k)=this->operator()(i,j,k)-a(i,j,k);
			}
			return res;
		}

		Tensor3d<T>&operator-=(const Tensor3d<T>&a){;
			if(this->c==a.c&&this->h==a.h&&this->l==a.l){
				for(size_t i=0;i<c;i++)
					for(size_t j=0;j<h;j++)
						for(size_t k=0;k<l;k++)
							this->operator()(i,j,k)-=a(i,j,k);
			}
			return*this;
		}

		Tensor3d<T>operator*(const T&a)const{
			Tensor3d<T>res(c,h,l);
			for(size_t i=0;i<c;i++)
				for(size_t j=0;j<h;j++)
					for(size_t k=0;k<l;k++)
						res(i,j,k)=this->operator()(i,j,k)*a;
			return res;
		}

		Tensor3d<T>&operator*=(const T&a){
			for(size_t i=0;i<c;i++)
				for(size_t j=0;j<h;j++)
					for(size_t k=0;k<l;k++)
						this->operator()(i,j,k)*=a;
			return*this;
		}

		friend Tensor3d<T>operator*(const T&a,const Tensor3d<T>&b){
			return b*a;
		}

		Tensor3d<T>hadamard(const Tensor3d<T>&a)const{
			Tensor3d<T>res;
			if(this->c==a.c&&this->h==a.h&&this->l==a.l){
				res.resize(c,h,l);
				for(size_t i=0;i<c;i++)
					for(size_t j=0;j<h;j++)
						for(size_t k=0;k<l;k++)
							res(i,j,k)=this->operator()(i,j,k)*a(i,j,k);
			}
			return res;
		}

		Matrix<T>flatten(){
			Matrix<T>res;
			res.v.resize(c*h*l);
			res.h=c*h*l;
			res.l=1;
			for(size_t i=0;i<v.size();i++)res.v[i]=this->v[i];
			return res;
		}

		static Tensor3d<T>deflatten(const Matrix<T>&a,size_t c,size_t h,size_t l){
			Tensor3d<T>res(c,h,l);
			res.v=a.v;
			return res;
		}

		Matrix<T>convolution(const Tensor3d<T>&a,size_t stride,size_t padding){
			Matrix<T>res((h+2*padding-a.h)/stride+1,(l+2*padding-a.l)/stride+1);
			for(size_t r_x=0;r_x<res.h;r_x++)
				for(size_t r_y=0;r_y<res.l;r_y++){
					res(r_x,r_y)=T(0);
					size_t x=r_x*stride;
					size_t y=r_y*stride;
					for(size_t i=0;i<a.c;i++)
						for(size_t j=0;j<a.h;j++)
							for(size_t k=0;k<a.l;k++){
								if(
									!(
										x+j-padding<0||
										y+k-padding<0||
										x+j-padding>=this->h||
										y+k-padding>=this->l
									)
								){
									res(r_x,r_y)+=
										this->operator()(i,x+j-padding,y+k-padding)*
										a(i,j,k);
								}
							}
				}
			return res;
		}

		static void con_for(
			Tensor3d<T>&res,
			const Tensor4d<T>&a,
			size_t stride,
			size_t padding,
			const Tensor3d<T>&b,
			size_t from,
			size_t to
		){
			for(size_t o_c=from;o_c<to;o_c++)
				for(size_t r_x=0;r_x<res.h;r_x++)
					for(size_t r_y=0;r_y<res.l;r_y++){
						res(o_c,r_x,r_y)=T(0);
						size_t x=r_x*stride;
						size_t y=r_y*stride;
						for(size_t i=0;i<a[o_c].c;i++)
							for(size_t j=0;j<a[o_c].h;j++)
								for(size_t k=0;k<a[o_c].l;k++){
									if(
										!(
											x+j-padding<0||
											y+k-padding<0||
											x+j-padding>=b.h||
											y+k-padding>=b.l
										)
									){
										res(o_c,r_x,r_y)+=
											b(i,x+j-padding,y+k-padding)*
											a[o_c](i,j,k);
									}
								}
					}
		}

		Tensor3d<T>convolution(const Tensor4d<T>&a,size_t stride,size_t padding)const{
			Tensor3d<T>res(
				a.size(),
				(h+2*padding-a[0].h)/stride+1,
				(l+2*padding-a[0].l)/stride+1
			);
			if(thread_num==0&&!over_threshold(200000,{c,h,l,a[0].c,a[0].h,a[0].l}))
				for(size_t o_c=0;o_c<a.size();o_c++)
					for(size_t r_x=0;r_x<res.h;r_x++)
						for(size_t r_y=0;r_y<res.l;r_y++){
							res(o_c,r_x,r_y)=T(0);
							size_t x=r_x*stride;
							size_t y=r_y*stride;
							for(size_t i=0;i<a[o_c].c;i++)
								for(size_t j=0;j<a[o_c].h;j++)
									for(size_t k=0;k<a[o_c].l;k++){
										if(
											!(
												(long long)(x+j)-
												(long long)(padding)<
												0||
												(long long)(y+k)-
												(long long)(padding)<
												0||
												(long long)(x+j)-
												(long long)(padding)>=
												(long long)this->h||
												(long long)(y+k)-
												(long long)(padding)>=
												(long long)this->l
											)
										){
											res(o_c,r_x,r_y)+=
												this->operator()(
													i,
													x+j-padding,
													y+k-padding
												)*
												a[o_c](i,j,k);
										}
									}
						}
			else{
				std::vector<std::thread>ts;
				size_t i=0;
				size_t real_thread_num=std::min(
					static_cast<size_t>(thread_num),
					a.size()
				);
				if(real_thread_num==0)real_thread_num=1;
				size_t l=(a.size()+real_thread_num-1)/real_thread_num;
				while(i<a.size()){
					if(i+l<a.size())
						ts.push_back(
							std::thread(
								con_for,
								std::ref(res),
								std::cref(a),
								stride,
								padding,
								std::cref(*this),
								i,
								i+l
							)
						);
					else
						ts.push_back(
							std::thread(
								con_for,
								std::ref(res),
								std::cref(a),
								stride,
								padding,
								std::cref(*this),
								i,
								a.size()
							)
						);
					i+=l;
				}
				for(size_t i=0;i<ts.size();i++)ts[i].join();
			}
			return res;
		}
	};
}

#endif
