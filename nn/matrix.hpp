#ifndef MATRIX_HPP
#define MATRIX_HPP

#include<vector>
#include<iostream>
#include<concepts>

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
            for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res(j,i)=this->operator()(i,j);
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
                for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res(i,j)=this->operator()(i,j)+a(i,j);
            }
            return res;
        }

        Matrix<T>&operator+=(const Matrix&a){
            if(this->h==a.h&&this->l==a.l)for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)this->operator()(i,j)+=a(i,j);
            return*this;
        }

        Matrix<T>operator-(const Matrix<T>&a)const{
            Matrix<T>res;
            if(this->h==a.h&&this->l==a.l){
                res.resize(h,l);
                for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res(i,j)=this->operator()(i,j)-a(i,j);
            }
            return res;
        }

        Matrix<T>&operator-=(const Matrix&a){
            if(this->h==a.h&&this->l==a.l)for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)this->operator()(i,j)-=a(i,j);
            return*this;
        }

        Matrix<T>operator*(const T&a)const{
            Matrix<T>res(h,l);
            for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res(i,j)=this->operator()(i,j)*a;
            return res;
        }

        Matrix<T>&operator*=(const T&a){
            for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)this->operator()(i,j)*=a;
            return*this;
        }

        friend Matrix<T>operator*(const T&a,const Matrix<T>&b){
            Matrix<T>res(b.h,b.l);
            for(size_t i=0;i<b.h;i++)for(size_t j=0;j<b.l;j++)res(i,j)=b(i,j)*a;
            return res;
        }

        Matrix<T>operator*(const Matrix<T>&a)const{
            Matrix<T>res;
            if(this->l==a.h){
                res.resize(this->h,a.l);
                for(size_t i=0;i<res.h;i++)for(size_t j=0;j<res.l;j++){
                    res(i,j)=T{};
                    for(size_t k=0;k<this->l;k++)res(i,j)+=(this->operator()(i,k)*a(k,j));
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
                for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res(i,j)=this->operator()(i,j)*a(i,j);
            }
            return res;
        }
    };
}

#endif