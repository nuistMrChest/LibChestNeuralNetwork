#ifndef MATRIX_HPP
#define MATRIX_HPP

#include<vector>
#include<iostream>
#include<concepts>
#include<thread>
#include<functional>

#ifndef thread_num
#define thread_num 10
#endif

namespace LibCN{
	template<typename T>concept Element=requires(T a,T b,std::iostream&os){
		{a+b}->std::same_as<T>;
		{a+=b}->std::same_as<T&>;
		{a-b}->std::same_as<T>;
		{a-=b}->std::same_as<T&>;
		{a*b}->std::same_as<T>;
		{a*=b}->std::same_as<T&>;
		{os<<a}->std::same_as<std::ostream&>;
		{a>b}->std::same_as<bool>;
		{a<b}->std::same_as<bool>;
		{a>=b}->std::same_as<bool>;
		{a<=b}->std::same_as<bool>;
		{a==b}->std::same_as<bool>;
		{a!=b}->std::same_as<bool>;
		{a/b}->std::same_as<T>;
		T{};
		T{0};
		T();
		T(0);
	};

	template<Element T>struct Matrix{
		std::vector<T>v;
		size_t h,l;

		Matrix(){
		h=0;
			l=0;
			v.resize(0);
		}

		Matrix(size_t h,size_t l){
			this->h=h;
			this->l=l;
			v.resize(h*l);
		}

		Matrix(std::vector<std::vector<T>>&a){
			this->h=a.size();
			this->l=a[0].size();
			this->v.resize(0);
			for(size_t i=0;i<h;i++){
				for(size_t j=0;j<l;j++){
					v.push_back(a[i][j]);
				}
			}
		}

		Matrix(const Matrix<T>&a){
			this->h=a.h;
			this->l=a.l;
			this->v=a.v;
		}

		Matrix<T>&operator=(const Matrix<T>&a){
			this->h=a.h;
			this->l=a.l;
			this->v=a.v;
			return*this;
		}

		Matrix(std::initializer_list<std::initializer_list<T>>init){
			h=init.size();
			l=init.begin()->size();
			v.reserve(h*l);
			for(const auto&row:init){
				for(const auto&x:row){
					v.push_back(x);
				}
			}
		}

		T&operator()(size_t i,size_t j){
			return v[i*l+j];
		}
		
		const T&operator()(size_t i,size_t j)const{
			return v[i*l+j];
		}

		friend std::ostream&operator<<(std::ostream&os,const Matrix<T>&a){
			if(a.h==0)os<<"{ NULL }";
			for(size_t i=0;i<a.h;i++){
				if(i==0)os<<"{";
				else os<<" ";
				os<<" ";
				for(size_t j=0;j<a.l;j++){
					os<<a(i,j)<<" ";
				}
				if(i==a.h-1)os<<"}";
				else os<<"\n";
			}
			return os;
		}

		Matrix<T>transpose()const{
			Matrix<T>res(l,h);
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<l;j++)
					res(j,i)=this->operator()(i,j);
			return res;
		}

		void resize(size_t h,size_t l){
			this->h=h;
			this->l=l;
			this->v.resize(h*l);
		}

		Matrix<T>operator+(const Matrix<T>&a)const{
			Matrix<T>res;
			if(this->h==a.h&&this->l==a.l){
				res.resize(h,l);
				for(size_t i=0;i<h;i++)
					for(size_t j=0;j<l;j++)
						res(i,j)=this->operator()(i,j)+a(i,j);
			}
			return res;
		}

		Matrix<T>&operator+=(const Matrix&a){
			if(this->h==a.h&&this->l==a.l)
				for(size_t i=0;i<h;i++)
					for(size_t j=0;j<l;j++)
						this->operator()(i,j)+=a(i,j);
			return*this;
		}

		Matrix<T>operator-(const Matrix<T>&a)const{
			Matrix<T>res;
			if(this->h==a.h&&this->l==a.l){
				res.resize(h,l);
				for(size_t i=0;i<h;i++)
					for(size_t j=0;j<l;j++)
						res(i,j)=this->operator()(i,j)-a(i,j);
			}
			return res;
		}

		Matrix<T>&operator-=(const Matrix&a){
			if(this->h==a.h&&this->l==a.l)
				for(size_t i=0;i<h;i++)
					for(size_t j=0;j<l;j++)this->operator()(i,j)-=a(i,j);
			return*this;
		}

		Matrix<T>operator*(const T&a)const{
			Matrix<T>res(h,l);
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<l;j++)
					res(i,j)=this->operator()(i,j)*a;
			return res;
		}

		Matrix<T>&operator*=(const T&a){
			for(size_t i=0;i<h;i++)
				for(size_t j=0;j<l;j++)
					this->operator()(i,j)*=a;
			return*this;
		}

		friend Matrix<T>operator*(const T&a,const Matrix<T>&b){
			Matrix<T>res(b.h,b.l);
			for(size_t i=0;i<b.h;i++)
				for(size_t j=0;j<b.l;j++)
					res(i,j)=b(i,j)*a;
			return res;
		}

		static void subMatrixMultiplication(
			size_t f,
			size_t m,
			size_t n,
			size_t p,
			Matrix<T>&res,
			const Matrix<T>&a,
			const Matrix<T>&b
		){
			for(size_t i=f;i<m;i++)for(size_t j=0;j<p;j++){
				res(i,j)=T(0);
				for(size_t k=0;k<n;k++){
					res(i,j)+=a(i,k)*b(k,j);
				}
			}
		}

		Matrix<T>operator*(const Matrix<T>&a)const{
			const Matrix<T>&b=*this;
			Matrix<T>res;
			if(this->l==a.h){
				size_t m=this->h;
				size_t n=this->l;
				size_t p=a.l;
				if(m<((200000/p)/n)||thread_num==0){
					res.resize(this->h,a.l);
					for(size_t i=0;i<res.h;i++)for(size_t j=0;j<res.l;j++){
						res(i,j)=T{};
						for(size_t k=0;k<this->l;k++)
							res(i,j)+=(this->operator()(i,k)*a(k,j));
					}
				}
				else{
					res.resize(this->h,a.l);
					for(size_t i=0;i<res.v.size();i++)res.v[i]=T(0);
					size_t i=0;
					size_t l=std::min(m,m/thread_num);
					std::vector<std::thread>ts;
					while(i<m){
						if(i+l<m)ts.push_back(
							std::thread(
								subMatrixMultiplication,
								i,
								i+l,
								n,
								p,
								std::ref(res),
								std::cref(b),
								std::cref(a)
							)
						);
						else ts.push_back(
							std::thread(
								subMatrixMultiplication,
								i,
								m,
								n,
								p,
								std::ref(res),
								std::cref(b),
								std::cref(a)
							)
						);
						i+=l;
					}
					for(size_t i=0;i<ts.size();i++)ts[i].join();
				}
			}
			return res;
		}

		Matrix<T>&operator*=(const Matrix<T>&a){
			*this=*this*a;
			return*this;
		}
		
		Matrix<T>hadamard(const Matrix<T>&a)const{
			Matrix<T>res;
			if(this->l==a.l&&this->h==a.h){
				res.resize(h,l);
				for(size_t i=0;i<h;i++)
					for(size_t j=0;j<l;j++)
						res(i,j)=this->operator()(i,j)*a(i,j);
			}
			return res;
		}
	};

	template<typename T>using Tensor2d=Matrix<T>;
}

#endif
